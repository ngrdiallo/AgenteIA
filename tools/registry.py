"""
Tool Registry: whitelist di strumenti eseguibili dall'agente.

V2: ZERO eval() — solo execute_tool() da whitelist.
Ogni tool deve essere registrato esplicitamente con JSON schema.
"""
import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


@dataclass
class ToolDefinition:
    """Definizione di un tool eseguibile."""
    name: str
    description: str
    json_schema: dict
    func: Callable
    requires_context: bool = True  # Se True, richiede HybridSearchEngine


class ToolRegistry:
    """
    Registry per tool eseguibili in modo sicuro.
    V2: NO eval() — solo whitelist.
    """
    
    def __init__(self):
        self._tools: dict[str, ToolDefinition] = {}
        self._hybrid_search_engine: Optional[Any] = None
    
    def set_hybrid_search(self, engine):
        """Imposta l'istanza di HybridSearchEngine per i tool che la usano."""
        self._hybrid_search_engine = engine
    
    def register(self, tool: ToolDefinition):
        """Registra un tool nella whitelist."""
        self._tools[tool.name] = tool
        logger.info(f"🔧 Registered tool: {tool.name}")
    
    def get_tool(self, name: str) -> Optional[ToolDefinition]:
        """Ritorna tool per nome."""
        return self._tools.get(name)
    
    def list_tools(self) -> list[dict]:
        """Lista tutti i tool disponibili."""
        return [
            {
                "name": t.name,
                "description": t.description,
                "parameters": t.json_schema,
            }
            for t in self._tools.values()
        ]
    
    def get_openai_tools(self) -> list[dict]:
        """Ritorna format OpenAI per tool_calling."""
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.json_schema,
                }
            }
            for t in self._tools.values()
        ]
    
    async def execute_tool(self, name: str, arguments: dict, context: Optional[dict] = None) -> Any:
        """
        Esegue un tool dalla whitelist.
        
        Args:
            name: nome del tool
            arguments: dict con argomenti
            context: contesto aggiuntivo (collection_id, etc.)
        
        Returns:
            Risultato dell'esecuzione
        
        Raises:
            ValueError: se tool non trovato o non registrato
        """
        tool = self._tools.get(name)
        if not tool:
            raise ValueError(f"Tool '{name}' not found in registry")
        
        logger.info(f"🔧 Executing tool: {name} with args: {arguments}")
        
        try:
            # Prepara kwargs
            kwargs = {**arguments}
            if tool.requires_context and context:
                kwargs["context"] = context
            if tool.requires_context and self._hybrid_search_engine:
                kwargs["search_engine"] = self._hybrid_search_engine
            
            # Esegui tool (sync o async)
            if asyncio.iscoroutinefunction(tool.func):
                result = await tool.func(**kwargs)
            else:
                result = tool.func(**kwargs)
            
            logger.info(f"✅ Tool {name} completed")
            return result
            
        except Exception as e:
            logger.error(f"❌ Tool {name} failed: {e}")
            raise


# ============================================================
# TOOL IMPLEMENTATIONS
# ============================================================

async def retrieve_docs(
    query: str,
    collection_id: str = "default",
    top_k: int = 5,
    search_engine: Optional[Any] = None,
    context: Optional[dict] = None
) -> list[dict]:
    """
    Recupera documenti rilevanti dalla knowledge base.
    """
    if search_engine is None:
        return [{"error": "Search engine not available"}]
    
    try:
        results = await asyncio.to_thread(
            search_engine.search,
            query=query,
            collection_name=collection_id,
            top_k=top_k,
        )
        
        return [
            {
                "chunk_id": r.chunk_id,
                "text": r.text[:500],  # Truncate per evitare token waste
                "score": round(r.score, 3),
                "source": r.source_file,
                "page": r.page_number,
            }
            for r in results
        ]
    except Exception as e:
        logger.error(f"retrieve_docs failed: {e}")
        return [{"error": str(e)}]


def get_document_metadata(
    document_id: str,
    context: Optional[dict] = None
) -> dict:
    """
    Recupera metadati di un documento specifico.
    """
    # Placeholder - implementare con indexer.get()
    return {
        "document_id": document_id,
        "title": "Documento",
        "pages": 0,
        "status": "not_implemented",
    }


async def search_web(
    query: str,
    max_results: int = 3
) -> list[dict]:
    """
    Cerca sul web usando Tavily (fallback: DuckDuckGo).
    """
    try:
        from tavily import TavilyClient
        from config import settings
        
        api_key = getattr(settings, 'TAVILY_API_KEY', None)
        if not api_key:
            return [{"error": "Tavily API key not configured"}]
        
        client = TavilyClient(api_key=api_key)
        results = client.search(query=query, max_results=max_results)
        
        return [
            {
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "content": r.get("content", "")[:300],
            }
            for r in results.get("results", [])
        ]
    except Exception as e:
        logger.error(f"search_web failed: {e}")
        return [{"error": str(e)}]


def summarize_text(
    text: str,
    max_length: int = 200
) -> dict:
    """
    Summarizza un testo (placeholder - richiede LLM).
    """
    words = text.split()
    if len(words) <= max_length:
        return {"summary": text, "original_length": len(words)}
    
    summary = " ".join(words[:max_length]) + "..."
    return {
        "summary": summary,
        "original_length": len(words),
        "truncated": True,
    }


def analyze_image(
    image_path: str,
    question: str = "Descrivi questa immagine"
) -> dict:
    """
    Analizza un'immagine (placeholder - richiede vision-capable LLM).
    """
    return {
        "image_path": image_path,
        "question": question,
        "result": "Vision analysis not implemented - requires GPT-4V or Gemini",
    }


# ============================================================
# REGISTRY SETUP
# ============================================================

def create_registry() -> ToolRegistry:
    """Crea e popola il registry con i tool."""
    registry = ToolRegistry()
    
    # retrieve_docs
    registry.register(ToolDefinition(
        name="retrieve_docs",
        description="Recupera documenti rilevanti dalla knowledge base",
        json_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Query di ricerca"},
                "collection_id": {"type": "string", "description": "ID collezione"},
                "top_k": {"type": "integer", "description": "Numero risultati", "default": 5},
            },
            "required": ["query"]
        },
        func=retrieve_docs,
        requires_context=True
    ))
    
    # get_document_metadata
    registry.register(ToolDefinition(
        name="get_document_metadata",
        description="Recupera metadati di un documento specifico",
        json_schema={
            "type": "object",
            "properties": {
                "document_id": {"type": "string", "description": "ID documento"},
            },
            "required": ["document_id"]
        },
        func=get_document_metadata,
        requires_context=False
    ))
    
    # search_web
    registry.register(ToolDefinition(
        name="search_web",
        description="Cerca informazioni sul web",
        json_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Query di ricerca"},
                "max_results": {"type": "integer", "description": "Max risultati", "default": 3},
            },
            "required": ["query"]
        },
        func=search_web,
        requires_context=False
    ))
    
    # summarize_text
    registry.register(ToolDefinition(
        name="summarize_text",
        description="Summarizza un testo",
        json_schema={
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Testo da summarizzare"},
                "max_length": {"type": "integer", "description": "Lunghezza massima", "default": 200},
            },
            "required": ["text"]
        },
        func=summarize_text,
        requires_context=False
    ))
    
    # analyze_image
    registry.register(ToolDefinition(
        name="analyze_image",
        description="Analizza un'immagine",
        json_schema={
            "type": "object",
            "properties": {
                "image_path": {"type": "string", "description": "Path immagine"},
                "question": {"type": "string", "description": "Domanda sull'immagine"},
            },
            "required": ["image_path"]
        },
        func=analyze_image,
        requires_context=False
    ))
    
    return registry


# Global registry
_tool_registry: Optional[ToolRegistry] = None


def get_tool_registry() -> ToolRegistry:
    """Ritorna l'istanza globale del registry."""
    global _tool_registry
    if _tool_registry is None:
        _tool_registry = create_registry()
    return _tool_registry


# SMOKE TEST
if __name__ == "__main__":
    print("=== Tool Registry ===")
    registry = create_registry()
    tools = registry.list_tools()
    print(f"Registered tools: {len(tools)}")
    for t in tools:
        print(f"  - {t['name']}: {t['description']}")
    
    # Test OpenAI format
    print("\nOpenAI tools format:")
    print(json.dumps(registry.get_openai_tools()[:2], indent=2))
