# Experiment 3: Half Adder & Full Adder using McCulloch-Pitts Neuron

def mp_neuron(inputs, weights, theta):
    return 1 if sum(x * w for x, w in zip(inputs, weights)) >= theta else 0

def half_adder(a, b):
    carry = mp_neuron([a, b], [1, 1], 2)          # A AND B
    z1 = mp_neuron([a, b], [1, -1], 1)            # A AND NOT B
    z2 = mp_neuron([a, b], [-1, 1], 1)            # NOT A AND B
    sum_out = mp_neuron([z1, z2], [1, 1], 1)      # z1 OR z2
    return sum_out, carry

def full_adder(a, b, cin):
    s1, c1 = half_adder(a, b)
    sum_final, c2 = half_adder(s1, cin)
    cout = mp_neuron([c1, c2], [1, 1], 1)         # c1 OR c2
    return sum_final, cout

# Interactive User Input
a = int(input("Enter A (0 or 1): "))
b = int(input("Enter B (0 or 1): "))
cin = int(input("Enter Cin (0 or 1): "))

h_s, h_c = half_adder(a, b)
f_s, f_c = full_adder(a, b, cin)

print(f"\nHalf Adder Result: Sum = {h_s}, Carry = {h_c}")
print(f"Full Adder Result: Sum = {f_s}, Cout = {f_c}")

print("\n--- Full Adder Truth Table ---")
print("A B Cin | Sum Cout")
for x in (0, 1):
    for y in (0, 1):
        for z in (0, 1):
            s, c = full_adder(x, y, z)
            print(f"{x} {y}  {z}  |  {s}    {c}")
