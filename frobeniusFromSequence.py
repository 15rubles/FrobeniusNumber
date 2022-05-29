import math


def Sequence(input_numbers):

    nums = input_numbers.copy()

    divisors = []
    addToFrobenius = 0

    def Multiplier(divs):
        multiplier = 1
        for div in divs:
            multiplier *= div
        return multiplier

    # make numbers coprime
    for i in range(3):
        for j in range(3):
            if i != j:
                divider = math.gcd(nums[i], nums[j])
                if divider > 1:
                    nums[i] //= divider
                    nums[j] //= divider
                    addToFrobenius += Multiplier(divisors) * nums[3 - i - j] * (divider - 1)
                    divisors.append(divider)
    print("divided nums", nums)
    # sort numbers
    nums.sort(reverse=True)
    input_numbers.sort(reverse=True)
    s = [nums[0]]
    # sort numbers
    nums.sort(reverse=True)
    # find S[0]
    s0 = 1
    while (s0 * nums[1]) % nums[0] != nums[2]:
        s0 += 1
    s.append(s0)
    q = []
    i = 0
    while s[-1] != 0:
        i += 1
        next_q = 1
        while s[i]*next_q < s[i-1]:
            next_q += 1
        q.append(next_q)
        s.append(next_q*s[i] - s[i-1])
    print("s", s)
    print("q", q)
    p = [0, 1]
    i = 2
    for q_el in q:
        p.append(q_el*p[i-1]-p[i-2])
        i += 1
    print("p", p)
    s_divided_by_q = []
    s_divided_by_q_print = []
    for i in range(len(s)):
        if p[i] == 0:
            s_divided_by_q.append(float('inf'))
        else:
            s_divided_by_q.append(s[i]/p[i])
            s_divided_by_q_print.append(str(s[i])+"/"+str(p[i]))
    s_divided_by_q_print.reverse()
    print(s_divided_by_q)
    print("Print", s_divided_by_q_print)
    c_divided_by_b = nums[2] / nums[1]
    u = 0
    while not (s_divided_by_q[u] > c_divided_by_b >= s_divided_by_q[u + 1]):
        u += 1
    print("u", u)

    frobeniusNumber = -nums[0] + nums[1]*(s[u]-1)+nums[2]*(p[u+1]-1)-min(nums[1]*s[u+1], nums[2]*p[u])
    print("Frobenius number:", frobeniusNumber)
    frobeniusNumber_with_input_nums = frobeniusNumber * Multiplier(divisors) + addToFrobenius
    print("Frobenius number with input nums:", frobeniusNumber_with_input_nums)
    output = []
    SUB = str.maketrans("0123456789-i+mu", "₀₁₂₃₄₅₆₇₈₉₋ᵢ₊ₘᵤ")
    output.append("Введенные числа: a="+str(input_numbers[0])+", b="+str(input_numbers[1])+", c="+str(input_numbers[2]))
    output.append("Упрощенные числа: a="+str(nums[0])+", b="+str(nums[1])+", c="+str(nums[2])+" (по формуле Джонсона)")
    output.append("")
    output.append("Формулы:")
    output.append("bS₀ ≡ c (mod a)")
    output.append("S-1".translate(SUB)+"= a")
    output.append("Si-2= qiSi-1".translate(SUB)+" - "+"Si".translate(SUB))
    output.append("P-1= ".translate(SUB)+"0"+"  P0= ".translate(SUB)+"1")
    output.append("Pi= qiPi-1".translate(SUB)+" - "+"Pi-2".translate(SUB))
    output.append("")
    output.append("Последовательности {Sᵢ}, {Pᵢ}, {qᵢ}:")
    for i in range(len(s)):
        convert = str(i-1)
        add = ''
        add += "S"+convert.translate(SUB)+"= "+str(s[i])+"  "
        add += "P"+convert.translate(SUB)+"= "+str(p[i])+"  "
        if i < len(q):
            add += "q"+str(i+1).translate(SUB)+"= "+str(q[i])
        output.append(add)
    output.append("Неравенство:")
    output.append("0=Sm+1/Pm+1<Sm/Pm<...<S0/P0<S-1/P-1=∞".translate(SUB))
    s_divided_by_q_print = '<'.join(map(str, s_divided_by_q_print))
    output.append("0="+s_divided_by_q_print+"<"+str(s[0])+"/"+str(p[0])+"=∞")
    output.append("c/b= "+ str(nums[2])+"/"+str(nums[1]))
    output.append("u= "+str(u))
    output.append("")
    output.append("Число Фробениуса для упрощенных чисел:")
    output.append("g(a,b,c)=-a+(Sᵤ-1)+c(P"+"u+1".translate(SUB)+"-1)-min{bS"+"u+1".translate(SUB)+", cPᵤ}")
    output.append("g("+str(nums[0])+","+str(nums[1])+","+str(nums[2])+")=-"+str(nums[0])+"+"+str(nums[1])+"*("+str(s[u])+"-1)"+"+"+str(nums[2])+"*("+str(p[u+1])+"-1)"+"-"+"min{"+str(nums[1])+"*"+str(s[u+1])+", "+str(nums[2])+"*"+str(p[u])+"}="+str(frobeniusNumber))
    output.append("Число Фробениуса для введенных чисел: "+str(frobeniusNumber_with_input_nums))
    return output

