n=input()
res=[]
n=[int(k) for k in n.split(',')]
count=0
res.append(f'{n[0]}->')
j=0
for i in range(len(n)-2):
    count+=1
    if n[i+1]-n[i]!=1:
        res[j]+=f'{n[i]}'
        j+=1
        res.append(f'{n[i+1]}->')
        count=0
    
if n[len(n)-1]-n[len(n)-2]==1:
    res[j]+=f'{n[len(n)-1]}'
else:
    res[j]+=f'{n[len(n)-2]}'
    res.append(f'{n[len(n)-1]}')
print(','.join(res))