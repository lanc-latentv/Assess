# -*- coding: utf-8 -*-
"""Untitled5.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1lY3K42czFPXMlq8kiw5yPr8KTJyEUHH0
"""

#1

a=int(input("enter the length: "))
b=int(input("enter the width: "))
area=a*b
print("The area is "+str(area)+"  m*m\n")

if area<100:
  print("small")
elif (area>100) and (area<200):
  print("medium")
elif area>200:
  print("large")

#2
h=int(input("enter the height: "))
w=int(input("enter the weight: "))
bmi=w/(h**2)

print("The BMI is "+str(bmi))

#3
def add(wh,wh_name):
      if wh_name not in wh:
        wh[wh_name] = {}

def update(wh,wh_name,p_name,mark):
      if p_name in wh[wh_name]:
        wh[wh_name][p_name] += mark
      else:
        wh[wh_name][p_name] = mark

def retrieve(wh, wh_name, p_name):
    if wh_name in wh and p_name in wh[wh_name]:
        return wh[wh_name][p_name]
    else:
        return 0

wh={}
while True:
  print("\n Management System")
  print("1. Add")
  print("2. Update")
  print("3. Retrieve ")
  print("4. exit ")

  choice = input("Enter your choice (1/2/3/4): ")

  if choice == '1':
    subject_name = input("Enter the subject name: ")
    add(wh,subject_name)
  elif choice == '2':
    subject_name = input("Enter the subject name: ")
    student_name = input("Enter the student name: ")
    mark = int(input("Enter the marks to add: "))
    update(wh, subject_name, student_name, mark)
    print("updated successfully!")
  elif choice == '3':
    subject_name = input("Enter the subject name: ")
    student_name = input("Enter the student name: ")
    mark = retrieve(wh,subject_name, student_name)
    print(f"Mark of {student_name} in {subject_name}: {mark}")
    print(wh)
  elif choice == '4':
    print("thank you!!!")
    break
  else:
    print("Invalid choice! Please enter 1, 2, or 3.")

#LAB 2
#4


a=int(input("Enter your age   "))


if a<10:
  print("Oops sorry no entry for children ")
elif (a==10) and (a<=18):
  print("welcome teen....here are your recommendations")
elif (a>=18) and (a<40):
  print("HI adult....here are your recommendations")
elif a>40:
  print("Hi senior....here are your recommendations")

#5


n=20   # number of subscriber id's can be mentioned here
for i in range(n):
  if(i%2==0):
    print(i)

#6

password="abcd"

while True:
  a=input("Enter password to enter  :  ")
  if a==password:
    print("The file is opened")
    break
  else:
    print("Sorry wrong password , enter password again")

#LAB 3
#7
print("5-very good")
print("4-good")
print("3-average ")
print("2-can do better")
print("1-poor")
n=5 # no of customers
s=0
a=[]
for i in range(n):
  a[i]=int(input("enter the score : "))

for j in range(n):
  s=s+a[j]

m=s/n #avg score

if m<50:
  print("poor")
elif m>100:
  print("average")
elif m>150:
  print("good")
elif m>50:
  print("very good")

#8

a=input("enter the word   ")
count=0

for i in a:
  if (i=='a') or (i=='e') or (i=='i') or (i=='o') or (i=='u'):
    count+=1

print("count of vowels is "+str(count))

#9

#LAB 4

#10

try:
  with open('/content/b.txt','r') as file:
      li=file.readlines()
      for l in li:
        print (l)

except:
  print("file not found")

#11
try:
  a=int(input("enter data to the poll"))

except:
  print("sorry enter numeric value")

#12
try:
  a=int(input("enter a number:"))
  b=int(input("enter a number to divide the previous number:"))
  c=a/b
  print("ans :"+str(c))
except:
  print("sorry not divisible")

#LAB 5
#13

with open('/content/c.txt','w') as file:
  x='Daily status reports\n'
  file.write(x)

#14

with open('/content/c.txt','r') as file:
  li=file.readlines()
  for l in li:
    print (l)

#15

with open('/content/c.txt','a') as file:
  y='Daily status reports are been updated\n'
  file.write(y)

with open('/content/c.txt','r') as file:
  li=file.readlines()
  for l in li:
    print (l)