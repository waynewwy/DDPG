def TransAngleto180(input):

    if (input > 180):
            input = input - 360
    if (input < -180):
            input = input + 360

    return input

def TransAngleto360(input):

    if (input > 360):

            input = input - 360
    if (input < -360):
            input = input + 360

    return  input