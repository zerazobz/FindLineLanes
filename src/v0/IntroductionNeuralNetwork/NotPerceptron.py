import pandas as pd

# TODO: Set weight1, weight2, and bias
weight1 = 0.0
weight2 = -2.0
bias = 1.0


# DON'T CHANGE ANYTHING BELOW
# Inputs and outputs
test_inputs = [(0, 0), (0, 1), (1, 0), (1, 1)]
correct_outputs = [True, False, True, False]
outputs = []

print("zip(test_inputs, correct_outputs)")
print(zip(test_inputs, correct_outputs))
print(list(zip(test_inputs, correct_outputs)))
print()

# Generate and check output
for test_input, correct_output in zip(test_inputs, correct_outputs):
    print("weight1: %d, test_input[0]: %d, weight2: %d, test_input[1]: %d, bias:%d" % (weight1, test_input[0], weight2, test_input[1], bias))
    linear_combination = weight1 * test_input[0] + weight2 * test_input[1] + bias
    print("\t linear_combination: %d" % (linear_combination))
    output = int(linear_combination >= 0)
    
    print("output [int(linear_combination >= 0)] are %d" % (output))
    print("correct_output are %d" % (correct_output))
    is_correct_string = 'Yes' if output == correct_output else 'No'
    outputs.append([test_input[0], test_input[1], linear_combination, output, is_correct_string])
    print()

# Print output
num_wrong = len([output[4] for output in outputs if output[4] == 'No'])
output_frame = pd.DataFrame(outputs, columns=['Input 1', '  Input 2', '  Linear Combination', '  Activation Output', '  Is Correct'])
if not num_wrong:
    print('Nice!  You got it all correct.\n')
else:
    print('You got {} wrong.  Keep trying!\n'.format(num_wrong))
print(output_frame.to_string(index=False))