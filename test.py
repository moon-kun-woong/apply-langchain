import keyword

print(keyword.kwlist)

print("\n")

if (keyword) : 
    print("ddddd"+ "\n")


def kkk(keyword):

    if not keyword:
        return print("이게 트라이."+ "\n")


kkk(False)

for i in [1,2,3,4]:
    print(i, end=", ")