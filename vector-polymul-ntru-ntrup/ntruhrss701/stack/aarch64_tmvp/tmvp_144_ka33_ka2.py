print('''#define _6_to_12(A00,A01,A10,A11,A20,A21,A30,A31,              \\
  A40,A41,A50,A51,T00,T01,T10,T11,P)                                    \\
  vst1q_u16((P)+8*12,(A30)); vst1q_u16((P)+8*13,(A31));                 \\
  vst1q_u16((P)+8*14,(A40)); vst1q_u16((P)+8*15,(A41));                 \\
  vst1q_u16((P)+8*16,(A20)); vst1q_u16((P)+8*17,(A21));                 \\
  vst1q_u16((P)+8*18,(A30)); vst1q_u16((P)+8*19,(A31));                 \\
  vst1q_u16((P)+8*20,(A10)); vst1q_u16((P)+8*21,(A11));                 \\
  vst1q_u16((P)+8*22,(A20)); vst1q_u16((P)+8*23,(A21));                 \\
  T00 = vsubq_u16((A40),(A20)); T01 = vsubq_u16((A41),(A21));           \\
  T10 = vsubq_u16((A50),(A30)); T11 = vsubq_u16((A51),(A31));           \\
  T00 = vsubq_u16((T00),(A30)); T01 = vsubq_u16((T01),(A31));           \\
  T10 = vsubq_u16((T10),(A40)); T11 = vsubq_u16((T11),(A41));           \\
  vst1q_u16((P)+8*0,(T00)); vst1q_u16((P)+8*1,(T01));                   \\
  vst1q_u16((P)+8*2,(T10)); vst1q_u16((P)+8*3,(T11));                   \\
  T00 = vsubq_u16((A20),(A10)); T01 = vsubq_u16((A21),(A11));           \\
  T10 = vsubq_u16((A30),(A20)); T11 = vsubq_u16((A31),(A21));           \\
  T00 = vsubq_u16((T00),(A30)); T01 = vsubq_u16((T01),(A31));           \\
  T10 = vsubq_u16((T10),(A40)); T11 = vsubq_u16((T11),(A41));           \\
  vst1q_u16((P)+8*4,(T00)); vst1q_u16((P)+8*5,(T01));                   \\
  vst1q_u16((P)+8*6,(T10)); vst1q_u16((P)+8*7,(T11));                   \\
  T00 = vsubq_u16((A00),(A10)); T01 = vsubq_u16((A01),(A11));           \\
  T10 = vsubq_u16((A10),(A20)); T11 = vsubq_u16((A11),(A21));           \\
  T00 = vsubq_u16((T00),(A20)); T01 = vsubq_u16((T01),(A21));           \\
  T10 = vsubq_u16((T10),(A30)); T11 = vsubq_u16((T11),(A31));           \\
  vst1q_u16((P)+8*8,(T00)); vst1q_u16((P)+8*9,(T01));                   \\
  vst1q_u16((P)+8*10,(T10)); vst1q_u16((P)+8*11,(T11));                 \\
  // expand six 16xu16 to twelve 16xu16

#define _3_to_6(B00,B01,B10,B11,B20,B21,T0,T1,P)                     \\
  vst1q_u16((P)+8*0,(B00)); vst1q_u16((P)+8*1,(B01));                   \\
  vst1q_u16((P)+8*2,(B10)); vst1q_u16((P)+8*3,(B11));                   \\
  vst1q_u16((P)+8*4,(B20)); vst1q_u16((P)+8*5,(B21));                   \\
  T0 = vaddq_u16((B00),(B10)); T1 = vaddq_u16((B01),(B11));             \\
  vst1q_u16((P)+8*6,(T0)); vst1q_u16((P)+8*7,(T1));                     \\
  T0 = vaddq_u16((B00),(B20)); T1 = vaddq_u16((B01),(B21));             \\
  vst1q_u16((P)+8*8,(T0)); vst1q_u16((P)+8*9,(T1));                     \\
  T0 = vaddq_u16((B10),(B20)); T1 = vaddq_u16((B11),(B21));             \\
  vst1q_u16((P)+8*10,(T0)); vst1q_u16((P)+8*11,(T1));                     \\
  // expand 3 16xu16 to six 16xu16
  


''')  

print('''void tmvp_144_ka33_ka2(uint16_t *VecB, uint16_t *restrict ToepA){
  uint16_t TmpB[576], TmpA[1152], *Y;
  uint16x8_t A[36], TA[36], B[18], TB[18], T[6], *X;
  int i;
''')

for i in range(24) : print("  A[%d]=vld1q_u16(ToepA+8*%d);" % (i,i))

print("  _6_to_12(",end="")
for i in range(6,18) : print("A[%d]" % (i),end=",")
print("T[0],T[1],T[2],T[3],TmpA+192*5);")

print("  _6_to_12(",end="")
for i in range(12,24) : print("A[%d]" % (i),end=",")
print("T[0],T[1],T[2],T[3],TmpA+192*4);")

for i in range(12) : print("  TA[%d]=vsubq_u16(A[%d],A[%d]);" % (i,i,i+6))
for i in range(12) : print("  TA[%d]=vsubq_u16(TA[%d],A[%d]);" % (i,i,i+12))

print("  _6_to_12(",end="")
for i in range(12) : print("TA[%d]" % (i),end=",")
print("T[0],T[1],T[2],T[3],TmpA+192*2);")

for i in range(24,30) : print("  A[%d]=vld1q_u16(ToepA+8*%d);" % (i,i))

print("  _6_to_12(",end="")
for i in range(18,30) : print("A[%d]" % (i),end=",")
print("T[0],T[1],T[2],T[3],TmpA+192*3);")

for i in range(12) : print("  TA[%d]=vsubq_u16(A[%d],A[%d]);" % (i,i+12,i+6))
for i in range(12) : print("  TA[%d]=vsubq_u16(TA[%d],A[%d]);" % (i,i,i+18))

print("  _6_to_12(",end="")
for i in range(12) : print("TA[%d]" % (i),end=",")
print("T[0],T[1],T[2],T[3],TmpA+192*1);")

for i in range(30,36) : print("  A[%d]=vld1q_u16(ToepA+8*%d);" % (i,i))

for i in range(12) : print("  TA[%d]=vsubq_u16(A[%d],A[%d]);" % (i,i+24,i+18))
for i in range(12) : print("  TA[%d]=vsubq_u16(TA[%d],A[%d]);" % (i,i,i+12))

print("  _6_to_12(",end="")
for i in range(12) : print("TA[%d]" % (i),end=",")
print("T[0],T[1],T[2],T[3],TmpA+192*0);")

print("  // TmpA set")

for i in range(18) :
    print("  B[%d]=vld1q_u16(VecB+8*%d);" % (i,i))

for i in range(3) :
    print("  _3_to_6(",end="")
    for j in range(6*i,6*i+6) : print("B[%d]" % (j),end=",")
    print("T[0],T[1],TmpB+96*%d);"%(i))

for i in range(6) : print("  TB[%d]=vaddq_u16(B[%d],B[%d]);" % (i,i,i+6))
for i in range(6) : print("  TB[%d]=vaddq_u16(B[%d],B[%d]);" % (i+6,i,i+12))
for i in range(6) : print("  TB[%d]=vaddq_u16(B[%d],B[%d]);" % (i+12,i+6,i+12))
    
for i in range(3) :
    print("  _3_to_6(",end="")
    for j in range(6*i,6*i+6) : print("TB[%d]" % (j),end=",")
    print("T[0],T[1],TmpB+96*%d);"%(i+3))

print('''  // TmpB set
  for(i=0; i<18; i++) tmvp_16x16_x2_ka(TmpB+32*i,TmpA+64*i,TmpB+32*i+16,TmpA+64*i+32);''')

for i in range(3) :
    for j in range(6) : print("  B[%d]=vld1q_u16(TmpB+8*%d);" % (6*i+j,12*i+j))
    for j in range(6) : print("  T[%d]=vld1q_u16(TmpB+8*%d);" % (j,12*i+j+6))
    for j in range(2) :
        print("  B[%d]=vaddq_u16(B[%d],T[%d]);" % (6*i+j,6*i+j,j))
        print("  B[%d]=vaddq_u16(B[%d],T[%d]);" % (6*i+j,6*i+j,j+2)) 
        print("  B[%d]=vaddq_u16(B[%d],T[%d]);" % (6*i+j+2,6*i+j+2,j))
        print("  B[%d]=vaddq_u16(B[%d],T[%d]);" % (6*i+j+2,6*i+j+2,j+4)) 
        print("  B[%d]=vaddq_u16(B[%d],T[%d]);" % (6*i+j+4,6*i+j+4,j+2))
        print("  B[%d]=vaddq_u16(B[%d],T[%d]);" % (6*i+j+4,6*i+j+4,j+4)) 
    
for i in range(3) :
    for j in range(6) : print("  TB[%d]=vld1q_u16(TmpB+8*%d);" % (6*i+j,12*i+j+36))
    for j in range(6) : print("  T[%d]=vld1q_u16(TmpB+8*%d);" % (j,12*i+j+42))
    for j in range(2) :
        print("  TB[%d]=vaddq_u16(TB[%d],T[%d]);" % (6*i+j,6*i+j,j))
        print("  TB[%d]=vaddq_u16(TB[%d],T[%d]);" % (6*i+j,6*i+j,j+2)) 
        print("  TB[%d]=vaddq_u16(TB[%d],T[%d]);" % (6*i+j+2,6*i+j+2,j))
        print("  TB[%d]=vaddq_u16(TB[%d],T[%d]);" % (6*i+j+2,6*i+j+2,j+4)) 
        print("  TB[%d]=vaddq_u16(TB[%d],T[%d]);" % (6*i+j+4,6*i+j+4,j+2))
        print("  TB[%d]=vaddq_u16(TB[%d],T[%d]);" % (6*i+j+4,6*i+j+4,j+4)) 
    if (i == 0) : k = [0,6]
    elif (i == 1) : k = [0,12]
    elif (i == 2) : k = [6,12]
    for j in range(6) : 
        print("  B[%d]=vaddq_u16(B[%d],TB[%d]);" % (k[0]+j,k[0]+j,6*i+j))
        print("  B[%d]=vaddq_u16(B[%d],TB[%d]);" % (k[1]+j,k[1]+j,6*i+j)) 
    
for i in range(9) :
    for j in range(2) :
        print("  vst1q_u16(VecB+%d,B[%d]);" % (16*i+8*j,16-2*i+j))

        
print("}")
