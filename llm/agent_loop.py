"""
Agent Loop: ReAct pattern per agenti LLM con tool calling.

V2: ZERO eval() — usa ToolRegistry per esecuzione sicura.
Pattern ReAct:
  1. Think: LLM analizza e decide next action
  2. Action: LLM chiama un tool (se necessario)
  3. Observe: Tool result viene aggiunto alla conversation
  4. Repeat fino a max_steps o risposta finale

Importante: Questo file deve rimanere ASYNC.
"""
import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from llm.backend_pool import BackendPool
from llm.capability_matrix import get_models_for_intent
from llm.response_cache import get_response_cache
from tools.registry import get_tool_registry, ToolRegistry

logger = logging.getLogger(__name__)


@dataclass
class AgentStep:
    """Singolo step nel loop dell'agente."""
    step_num: int
    thought: str
    action: Optional[str] = None
    action_input: Optional[dict] = None
    observation: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    latency_ms: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0


@dataclass
class AgentResult:
    """Risultato finale dell'agente."""
    answer: str
    steps_taken: int
    tools_called: list[str]
    completed: bool
    steps: list[AgentStep] = field(default_factory=list)


class AgentLoop:
    """
    Agent loop con ReAct pattern.
    V2: usa ToolRegistry invece di eval().
    """
    
    def __init__(
        self,
        backend_pool: BackendPool,
        tool_registry: ToolRegistry,
        response_cache,
        max_steps: int = 4,
    ):
        self.backend_pool = backend_pool
        self.tool_registry = tool_registry
        self.response_cache = response_cache
        self.max_steps = max_steps
    
    def _build_system_prompt(self, intent: str) -> str:
        """Costruisce system prompt per l'agente."""
        tools = self.tool_registry.list_tools()
        tools_desc = "\n".join([
            f"- {t['name']}: {t['description']}"
            for t in tools
        ])
        
        return f"""Sei un assistente AI per Accademia di Belle Arti di Bari.

Disponi dei seguenti strumenti:
{tools_desc}

Istruzioni:
1. Analizza la domanda dell'utente
2. Se puoi rispondere direttamente, fallo
3. Se hai bisogno di informazioni, usa uno strumento
4. Dopo ogni azione, analizza il risultato prima di continuare

Formatta la risposta in modo chiaro e utile."""

    async def run(
        self,
        query: str,
        intent: str = "focused",
        collection_id: str = "default",
        model: Optional[str] = None,
    ) -> AgentResult:
        """
        Esegue il loop dell'agente.
        
        Args:
            query: Domanda dell'utente
            intent: Tipo di intent (meta, greeting, focused, comprehensive)
            collection_id: ID collezione per retrieval
            model: Modello specifico (opzionale)
        
        Returns:
            AgentResult con risposta e log degli step
        """
        steps = []
        tools_called = []
        
        # Prepara messages
        messages = [
            {"role": "system", "content": self._build_system_prompt(intent)},
            {"role": "user", "content": query},
        ]
        
        # Get tools per model selection
        tools = self.tool_registry.get_openai_tools()
        
        # Get model (usa capability_matrix se non specificato)
        if not model:
            candidates = get_models_for_intent(intent)
            if candidates:
                model = candidates[0].model_id
            else:
                model = "llama-3.1-70b-versatile"  # fallback
        
        # Check cache first
        cached = await self.response_cache.get(intent, query, collection_id)
        if cached:
            return AgentResult(
                answer=cached,
                steps_taken=0,
                tools_called=[],
                completed=True,
                steps=[]
            )
        
        # Loop ReAct
        for step_num in range(1, self.max_steps + 1):
            step_start = time.time()
            
            # Call LLM
            response = await self._call_llm(
                model=model,
                messages=messages,
                tools=tools if step_num < self.max_steps else None,
            )
            
            step = AgentStep(
                step_num=step_num,
                thought="",
                latency_ms=(time.time() - step_start) * 1000,
                tokens_in=response.get("usage", {}).get("prompt_tokens", 0),
                tokens_out=response.get("usage", {}).get("completion_tokens", 0),
            )
            
            # Parse response
            choice = response.get("choices", [{}])[0]
            message = choice.get("message", {})
            content = message.get("content", "")
            step.thought = content
            
            # Check for tool calls
            tool_calls = message.get("tool_calls", [])
            if not tool_calls:
                # No tool call = final answer
                steps.append(step)
                
                # Cache if not comprehensive
                if intent != "comprehensive":
                    await self.response_cache.set(
                        intent, query, collection_id, content
                    )
                
                return AgentResult(
                    answer=content,
                    steps_taken=step_num,
                    tools_called=tools_called,
                    completed=True,
                    steps=steps
                )
            
            # Execute tool calls
            for tc in tool_calls:
                tool_name = tc.get("function", {}).get("name")
                tool_args_str = tc.get("function", {}).get("arguments", "{}")
                
                # V2: SAFE parsing - NO eval()
                try:
                    tool_args = json.loads(tool_args_str)
                except json.JSONDecodeError:
                    tool_args = {}
                
                step.action = tool_name
                step.action_input = tool_args
                tools_called.append(tool_name)
                
                # Execute tool via registry (SAFE)
                try:
                    context = {"collection_id": collection_id}
                    result = await self.tool_registry.execute_tool(
                        name=tool_name,
                        arguments=tool_args,
                        context=context,
                    )
                    
                    # Format result
                    if isinstance(result, (list, dict)):
                        result_str = json.dumps(result, ensure_ascii=False)
                    else:
                        result_str = str(result)
                    
                    step.observation = result_str[:500]  # Truncate
                    
                    # Add tool result to messages
                    messages.append({
                        "role": "assistant",
                        "content": content,
                        "tool_calls": [tc]
                    })
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.get("id", "unknown"),
                        "content": result_str
                    })
                    
                except Exception as e:
                    step.observation = f"Error: {str(e)}"
                    logger.error(f"Tool {tool_name} failed: {e}")
            
            steps.append(step)
            
            # Check for final answer in last step
            if step_num == self.max_steps:
                # Return last content as answer
                return AgentResult(
                    answer=content,
                    steps_taken=step_num,
                    tools_called=tools_called,
                    completed=False,  # Hit max_steps
                    steps=steps
                )
        
        # Should not reach here
        return AgentResult(
            answer="Max steps reached",
            steps_taken=self.max_steps,
            tools_called=tools_called,
            completed=False,
            steps=steps
        )
    
    async def _call_llm(
        self,
        model: str,
        messages: list,
        tools: Optional[list] = None,
    ) -> dict:
        """
        Chiama il LLM tramite BackendPool.
        Questo è un placeholder - l'implementazione dipende da come
        l'orchestratore esistente chiama i provider.
        """
        # Placeholder: in realtà chiamerà il provider adapter
        # Per ora ritorna una risposta dummy per testing
        return {
            "choices": [{
                "message": {
                    "content": "Questa è una risposta placeholder. L'integrazione con l'orchestratore esistente richiede ulteriore lavoro."
                }
            }],
            "usage": {"prompt_tokens": 100, "completion_tokens": 50},
        }


# ============================================================
# FACTORY
# ============================================================

def create_agent_loop(
    backend_pool: BackendPool,
    tool_registry: ToolRegistry,
    response_cache,
) -> AgentLoop:
    """Crea un'istanza di AgentLoop."""
    return AgentLoop(
        backend_pool=backend_pool,
        tool_registry=tool_registry,
        response_cache=response_cache,
        max_steps=4,
    )


# SMOKE TEST
if __name__ == "__main__":
    import asyncio
    
    async def test():
        print("=== Agent Loop Test ===")
        
        # Setup
        pool = BackendPool()
        registry = get_tool_registry()
        cache = get_response_cache()
        
        agent = create_agent_loop(pool, registry, cache)
        
        # Test run (without actual LLM call)
        result = await agent.run(
            query="Ciao, come stai?",
            intent="greeting",
        )
        
        print(f"Answer: {result.answer}")
        print(f"Steps: {result.steps_taken}")
        print(f"Tools: {result.tools_called}")
        print(f"Completed: {result.completed}")
    
    asyncio.run(test())
