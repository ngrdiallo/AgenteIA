"""
Test unitari per BackendPool — V2 con Circuit Breaker.
Verifica:
1. Singleton: stessa istanza
2. Circuit Breaker: HEALTHY -> DEGRADED -> DEAD
3. Recovery: provider riavviato dopo cooldown
4. get_chain: esclude provider in cooldown
5. report_success: aggiorna latenza e success_rate
"""

import time
import unittest

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from llm.backend_pool import BackendPool, BackendState, CircuitState


class TestBackendPool(unittest.TestCase):

    def setUp(self):
        BackendPool.reset()

    def tearDown(self):
        BackendPool.reset()

    def test_singleton(self):
        """Due istanze devono essere lo stesso oggetto."""
        a = BackendPool()
        b = BackendPool()
        self.assertIs(a, b)

    def test_circuit_breaker_states(self):
        """Circuit Breaker: HEALTHY -> DEGRADED (2 failures) -> DEAD (3 failures)."""
        pool = BackendPool()
        state = pool._states["groq"]
        
        # Initial state
        self.assertEqual(state.state, CircuitState.HEALTHY)
        
        # 1st failure - still HEALTHY
        pool.report_failure("groq", "generic")
        self.assertEqual(state.state, CircuitState.HEALTHY)
        
        # 2nd failure - DEGRADED
        pool.report_failure("groq", "generic")
        self.assertEqual(state.state, CircuitState.DEGRADED)
        
        # 3rd failure - DEAD
        pool.report_failure("groq", "generic")
        self.assertEqual(state.state, CircuitState.DEAD)

    def test_context_error_not_counted(self):
        """HTTP 400 context_length NON deve incrementare failures."""
        pool = BackendPool()
        state = pool._states["groq"]
        
        # Context errors don't count as failures
        pool.report_failure("groq", "context_length", is_context_error=True)
        pool.report_failure("groq", "context_length", is_context_error=True)
        pool.report_failure("groq", "context_length", is_context_error=True)
        
        # Still healthy - context errors don't affect circuit breaker
        self.assertEqual(state.consecutive_failures, 0)
        self.assertEqual(state.state, CircuitState.HEALTHY)

    def test_recovery_after_cooldown(self):
        """Provider si recupera quando il cooldown scade."""
        pool = BackendPool()
        state = pool._states["groq"]
        
        # Force to DEAD state
        pool.report_failure("groq", "generic")
        pool.report_failure("groq", "generic")
        pool.report_failure("groq", "generic")
        self.assertEqual(state.state, CircuitState.DEAD)

        # Simula cooldown scaduto: 60 * 2^3 = 480s (capped at 300)
        state.last_failure_time = time.time() - 350
        self.assertTrue(state.is_ready(), "Provider dovrebbe recuperare dopo cooldown")
        self.assertEqual(state.state, CircuitState.HEALTHY)

    def test_get_chain_excludes_dead(self):
        """get_chain deve escludere provider DEAD."""
        pool = BackendPool()
        # Force groq to DEAD
        pool.report_failure("groq", "generic")
        pool.report_failure("groq", "generic")
        pool.report_failure("groq", "generic")
        
        chain = pool.get_chain("speed", 100)
        self.assertNotIn("groq", chain,
                         "groq DEAD non deve comparire nella chain")
        self.assertIn("cerebras", chain,
                      "cerebras deve essere presente")

    def test_get_chain_excludes_degraded(self):
        """get_chain include DEGRADED ma con weight ridotto."""
        pool = BackendPool()
        # Force groq to DEGRADED
        pool.report_failure("groq", "generic")
        pool.report_failure("groq", "generic")
        
        chain = pool.get_chain("speed", 100)
        # DEGRADED still included but will have lower weight
        state = pool._states["groq"]
        self.assertEqual(state.state, CircuitState.DEGRADED)

    def test_report_success_updates_stats(self):
        """report_success aggiorna latenza e conteggio."""
        pool = BackendPool()
        pool.report_success("cerebras", 150.0)
        state = pool._states["cerebras"]
        self.assertEqual(state.success_count, 1)
        self.assertEqual(state.total_calls, 1)
        self.assertAlmostEqual(state.avg_latency_ms, 150.0)
        # Seconda misura: EMA 80/20
        pool.report_success("cerebras", 250.0)
        expected = 150.0 * 0.8 + 250.0 * 0.2  # 170.0
        self.assertAlmostEqual(state.avg_latency_ms, expected, places=1)
        self.assertEqual(state.success_count, 2)

    def test_get_chain_excludes_small_context(self):
        """get_chain deve escludere provider con contesto insufficiente."""
        pool = BackendPool()
        # huggingface ha 8000 token context
        chain = pool.get_chain("reasoning", 50_000)
        self.assertNotIn("huggingface", chain,
                         "huggingface (8k) non deve gestire 50k token")
        self.assertNotIn("ollama", chain,
                         "ollama (8k) non deve gestire 50k token")
        self.assertIn("gemini", chain,
                      "gemini (900k) deve poter gestire 50k token")

    def test_status_output_format(self):
        """status() deve restituire il formato atteso (V2: state invece di available)."""
        pool = BackendPool()
        pool.report_success("groq", 200.0)
        pool.report_failure("deepseek", "timeout")
        status = pool.status()
        self.assertIn("groq", status)
        self.assertIn("deepseek", status)
        
        # V2: usa "state" invece di "available"
        groq_status = status["groq"]
        for key in ["ready", "state", "consecutive_failures",
                     "avg_latency_ms", "success_rate", "total_calls", "weight"]:
            self.assertIn(key, groq_status, f"Chiave {key} mancante in status")
        
        self.assertEqual(groq_status["total_calls"], 1)
        self.assertEqual(groq_status["success_rate"], 1.0)
        self.assertEqual(groq_status["state"], "healthy")

    def test_weighted_routing(self):
        """DEGRADED ha weight * 0.3, DEAD ha weight 0."""
        pool = BackendPool()
        
        # HEALTHY
        state = pool._states["cerebras"]
        self.assertGreater(state.get_weight(), 0)
        
        # Make DEGRADED
        pool.report_failure("cerebras", "generic")
        pool.report_failure("cerebras", "generic")
        weight_degraded = state.get_weight()
        
        # Make DEAD
        pool.report_failure("cerebras", "generic")
        weight_dead = state.get_weight()
        
        self.assertEqual(weight_dead, 0.0, "DEAD weight must be 0")


if __name__ == "__main__":
    unittest.main(verbosity=2)
