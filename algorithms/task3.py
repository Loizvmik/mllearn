def solution(mas):
    length=len(mas)
    j,i=0,0
    while i <= length-1:
        if mas[i]!=0:
            tmp=mas[i]
            mas[i]=0
            mas[j]=tmp
            j+=1
        i+=1
    return mas

mas = [1,3,12]
mas=solution(mas)
print(mas)