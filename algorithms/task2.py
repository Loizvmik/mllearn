s = input().lower()

chars = []

for ch in s:
    if ch.isalnum():
        chars.append(ch)
        
length=len(chars)
point1=0
point2=length-1
flag=True
for i in range(length-1):
    if chars[point1]!=chars[point2]:
        flag=False
        break
    point1+=1
    point2-=1
if flag:
    print("YES")
else:    
    print("NO")