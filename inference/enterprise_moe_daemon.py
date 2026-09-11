# ==============================================================================
# ENTERPRISE LOCAL MOE DAEMON & KURAMOTO-NEWTON ROUTER
# ==============================================================================
# [AIR-GAPPED • ZERO-CLOUD DEPENDENCY • LOCAL SOCKET/HTTP ENDPOINT]
# ==============================================================================
import os
import sys
import time
import math
import json
import logging
import asyncio
from http.server import HTTPServer, BaseHTTPRequestHandler
from threading import Thread

# 1. Windows AsyncIO Event Loop Patch
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# 2. Air-Gap & Telemetry Lockdown
os.environ["CREWAI_TELEMETRY_OPT_OUT"] = "true"
os.environ["OTEL_SDK_DISABLED"] = "true"

# 3. Setup Local File Logging
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "enterprise_moe.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] (LocalDaemon) %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("MoELocalDaemon")

# ------------------------------------------------------------------------------
# LOCAL KURAMOTO-NEWTON VECTOR ROUTING ENGINE
# ------------------------------------------------------------------------------
class LocalKuramotoNewtonRouter:
    """Pure local vector phase-coupling router for MoE expert distribution."""
    def __init__(self, d_model: int = 128, num_experts: int = 16):
        self.d_model = d_model
        self.num_experts = num_experts
        
        # Initialize deterministic expert centroids and phases
        import random
        rng = random.Random(42)
        self.centroids = [
            [(rng.random() * 0.04 - 0.02) for _ in range(d_model)]
            for _ in range(num_experts)
        ]
        self.phases = [
            [(rng.random() * 2.0 * math.pi) for _ in range(d_model)]
            for _ in range(num_experts)
        ]
        self.expert_masses = [1.0 + (rng.random() * 0.5) for _ in range(num_experts)]
        logger.info(f"Initialized Local Kuramoto-Newton Engine ({num_experts} experts, d_model={d_model})")

    def route_token(self, token_vector=None, token_phase=None, top_k=2, G=1.2, K=2.5, gamma=2.0, epsilon=1e-3):
        import random
        rng = random.Random(int(time.time() * 1000) % 100000)
        
        if token_vector is None:
            token_vector = [(rng.random() * 2.0 - 1.0) for _ in range(self.d_model)]
        if token_phase is None:
            token_phase = [(rng.random() * 2.0 * math.pi) for _ in range(self.d_model)]

        # Token mass calculation
        token_mass = math.sqrt(sum(x * x for x in token_vector))

        # Order Parameter R
        sum_cos = sum(math.cos(p) for p in token_phase)
        sum_sin = sum(math.sin(p) for p in token_phase)
        order_r = math.sqrt(sum_cos**2 + sum_sin**2) / self.d_model

        scores = []
        for k in range(self.num_experts):
            r_sq = 0.0
            phase_sum = 0.0
            for d in range(self.d_model):
                diff = self.centroids[k][d] - token_vector[d]
                r_sq += diff * diff
                p_diff = token_phase[d] - self.phases[k][d]
                phase_sum += math.cos(p_diff)

            r = math.sqrt(r_sq + epsilon)
            f_newton = G * (token_mass * self.expert_masses[k]) / ((r ** gamma) + epsilon)
            f_kuramoto = K * (phase_sum / self.d_model)
            scores.append((f_newton + f_kuramoto, k))

        # Select Top-K Experts
        scores.sort(key=lambda x: x[0], reverse=True)
        top_selected = scores[:top_k]

        max_score = top_selected[0][0]
        exp_sum = sum(math.exp(score - max_score) for score, _ in top_selected)
        weights = [math.exp(score - max_score) / exp_sum for score, _ in top_selected]

        return {
            "selected_experts": [idx for _, idx in top_selected],
            "routing_weights": [round(w, 4) for w in weights],
            "order_parameter_R": round(order_r, 4),
            "max_coupled_force": round(max_score, 6)
        }

# Global Router Instance
router = LocalKuramotoNewtonRouter()

# ------------------------------------------------------------------------------
# LOCAL HTTP/REST ORCHESTRATION SERVER
# ------------------------------------------------------------------------------
class LocalDaemonRequestHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # Redirect to custom logger

    def _send_json_response(self, data, status=200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode("utf-8"))

    def do_GET(self):
        if self.path == "/health":
            self._send_json_response({
                "status": "HEALTHY",
                "mode": "AIR-GAPPED LOCAL",
                "engine": "Kuramoto-Newton MoE",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            })
        elif self.path == "/status":
            sample_route = router.route_token()
            self._send_json_response({
                "cluster_state": "ACTIVE",
                "latest_sample_routing": sample_route
            })
        else:
            self._send_json_response({"error": "Endpoint not found"}, status=404)

    def do_POST(self):
        if self.path == "/route":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length).decode("utf-8") if content_length > 0 else "{}"
            try:
                payload = json.loads(body) if body else {}
            except json.JSONDecodeError:
                payload = {}

            res = router.route_token(
                token_vector=payload.get("token_vector"),
                token_phase=payload.get("token_phase"),
                top_k=payload.get("top_k", 2)
            )
            logger.info(f"POST /route -> Selected Experts: {res['selected_experts']} | Weights: {res['routing_weights']}")
            self._send_json_response(res)
        else:
            self._send_json_response({"error": "Endpoint not found"}, status=404)

def run_http_server(host="127.0.0.1", port=8080):
    server = HTTPServer((host, port), LocalDaemonRequestHandler)
    logger.info(f"Local Daemon REST Server running at http://{host}:{port}/")
    server.serve_forever()

# ------------------------------------------------------------------------------
# MAIN LOCAL DAEMON LOOP
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    logger.info("==========================================================")
    logger.info("STARTING LOCAL ENTERPRISE MOE ORCHESTRATION DAEMON")
    logger.info("==========================================================")

    # Start HTTP REST server in background thread
    server_thread = Thread(target=run_http_server, daemon=True)
    server_thread.start()

    logger.info("Local Daemon active. Press Ctrl+C to terminate.")
    
    try:
        step = 0
        while True:
            step += 1
            res = router.route_token()
            if step % 10 == 0:
                logger.info(
                    f"[HEARTBEAT #{step}] Coherence R={res['order_parameter_R']} | "
                    f"Selected Experts={res['selected_experts']} | Weights={res['routing_weights']}"
                )
            time.sleep(1.0)
    except KeyboardInterrupt:
        logger.info("Shutting down Local Sovereign Daemon gracefully...")
        sys.exit(0)
