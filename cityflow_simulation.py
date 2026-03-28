import cityflow

engine = cityflow.Engine("config.json", thread_num=1)

tl_id = "intersection_1_1"

print("Controlling intersection:", tl_id)

for step in range(10):
    phase = step % 2
    engine.set_tl_phase(tl_id, phase)
    engine.next_step()
    print(f"Step {step + 1}, phase {phase}")