import numpy as np
import os

def num2str(num):
    """ parse num into hundredth palce string 123.45678900000 --> 123_45. works for numbers under 1000 and equal to or above 0.01

    Args:
        num (float): float to parse into string
    """
    if num < 0: return 'n' + num2str(-num)
    string = str(int(num)) + '_'
    num = np.round(num % 1, decimals=3)
    string += str(num)[2:5]
    return string

def flip_weights():
    vgds = ["2m", "4m", "8m", "16m"]
    localities = ["_local", "_global"]

    for vgd in vgds:
        for locality in localities:
            for file in os.listdir(f"{vgd}{locality}"):
                if file[0] == 'w':
                    ones = file.split("_")[1]
                    if ones[0] == 'n':
                        w = -float(ones[1:])
                        w -= 0.1 * float(file.split("_")[2])
                    else:
                        w = float(ones)
                        w += 0.1 * float(file.split("_")[2])
                    new_file = f"pf_{vgd}{locality}/w_{num2str(-w)}_{file.split("_")[3]}"

                    np.savetxt(new_file, np.loadtxt(f"{vgd}{locality}/{file}"), delimiter=',')  # Save the new weight


if __name__ == "__main__":
    flip_weights()  # Call the function