import os
import subprocess
import traci
import sumo

SUMO_HOME = sumo.SUMO_HOME
SUMO_CFG = os.path.join(
    SUMO_HOME,
    "tools",
    "game",
    "cross.sumocfg",
)

sumo_binary = os.path.join(SUMO_HOME, "bin", "sumo")

print("SUMO_HOME:", SUMO_HOME)
print("SUMO binary:", sumo_binary)
print("Config:", SUMO_CFG)

sumo_cmd = [sumo_binary, "-c", SUMO_CFG]

try:
    traci.start(
        sumo_cmd,
        numRetries=10,
        verbose=True,
        stdout=subprocess.PIPE,
    )
    print("Connected to SUMO")
    print("Traffic lights:", traci.trafficlight.getIDList())

    for step in range(10):
        traci.simulationStep()
        print(f"Step {step + 1} completed")

    traci.close()
    print("Simulation closed")

except Exception as e:
    print("TraCI start failed:")
    print(type(e).__name__, e)