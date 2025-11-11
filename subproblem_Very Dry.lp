\* Source Pyomo model name=unknown *\

max 
obj(Very_Dry):
+179.75520000000003 q(Very_Dry_25)
+182.1204 q(Very_Dry_26)
+184.4856 q(Very_Dry_27)
+186.85080000000002 q(Very_Dry_28)
+189.216 q(Very_Dry_29)
+191.58120000000002 q(Very_Dry_30)
+193.9464 q(Very_Dry_31)
+196.31160000000003 q(Very_Dry_32)
+198.67680000000004 q(Very_Dry_33)
+201.042 q(Very_Dry_34)
+203.40720000000002 q(Very_Dry_35)
+205.7724 q(Very_Dry_36)
+208.13760000000002 q(Very_Dry_37)
+210.50280000000004 q(Very_Dry_38)
+212.868 q(Very_Dry_39)
+215.2332 q(Very_Dry_40)
+217.5984 q(Very_Dry_41)
+219.9636 q(Very_Dry_42)
+222.32880000000003 q(Very_Dry_43)
+224.69400000000002 q(Very_Dry_44)
+227.05920000000003 q(Very_Dry_45)
+229.4244 q(Very_Dry_46)
+231.7896 q(Very_Dry_47)
+234.15480000000002 q(Very_Dry_48)
+52600.0 V(Very_Dry_48)

s.t.

c_e_V25(Very_Dry)_:
+0.0036 q(Very_Dry_25)
+1 V(Very_Dry_25)
= 4.300000000000001

c_e_V25(Dry)_:
+0.0036 q(Dry_25)
+1 V(Dry_25)
= 4.336

c_e_V25(Normal)_:
+0.0036 q(Normal_25)
+1 V(Normal_25)
= 4.372000000000001

c_e_V25(Wet)_:
+0.0036 q(Wet_25)
+1 V(Wet_25)
= 4.408

c_e_V25(Very_Wet)_:
+0.0036 q(Very_Wet_25)
+1 V(Very_Wet_25)
= 4.444000000000001

c_e_res_balance(Very_Dry_26)_:
+0.0036 q(Very_Dry_26)
-1 V(Very_Dry_25)
+1 V(Very_Dry_26)
= 0

c_e_res_balance(Very_Dry_27)_:
+0.0036 q(Very_Dry_27)
-1 V(Very_Dry_26)
+1 V(Very_Dry_27)
= 0

c_e_res_balance(Very_Dry_28)_:
+0.0036 q(Very_Dry_28)
-1 V(Very_Dry_27)
+1 V(Very_Dry_28)
= 0

c_e_res_balance(Very_Dry_29)_:
+0.0036 q(Very_Dry_29)
-1 V(Very_Dry_28)
+1 V(Very_Dry_29)
= 0

c_e_res_balance(Very_Dry_30)_:
+0.0036 q(Very_Dry_30)
-1 V(Very_Dry_29)
+1 V(Very_Dry_30)
= 0

c_e_res_balance(Very_Dry_31)_:
+0.0036 q(Very_Dry_31)
-1 V(Very_Dry_30)
+1 V(Very_Dry_31)
= 0

c_e_res_balance(Very_Dry_32)_:
+0.0036 q(Very_Dry_32)
-1 V(Very_Dry_31)
+1 V(Very_Dry_32)
= 0

c_e_res_balance(Very_Dry_33)_:
+0.0036 q(Very_Dry_33)
-1 V(Very_Dry_32)
+1 V(Very_Dry_33)
= 0

c_e_res_balance(Very_Dry_34)_:
+0.0036 q(Very_Dry_34)
-1 V(Very_Dry_33)
+1 V(Very_Dry_34)
= 0

c_e_res_balance(Very_Dry_35)_:
+0.0036 q(Very_Dry_35)
-1 V(Very_Dry_34)
+1 V(Very_Dry_35)
= 0

c_e_res_balance(Very_Dry_36)_:
+0.0036 q(Very_Dry_36)
-1 V(Very_Dry_35)
+1 V(Very_Dry_36)
= 0

c_e_res_balance(Very_Dry_37)_:
+0.0036 q(Very_Dry_37)
-1 V(Very_Dry_36)
+1 V(Very_Dry_37)
= 0

c_e_res_balance(Very_Dry_38)_:
+0.0036 q(Very_Dry_38)
-1 V(Very_Dry_37)
+1 V(Very_Dry_38)
= 0

c_e_res_balance(Very_Dry_39)_:
+0.0036 q(Very_Dry_39)
-1 V(Very_Dry_38)
+1 V(Very_Dry_39)
= 0

c_e_res_balance(Very_Dry_40)_:
+0.0036 q(Very_Dry_40)
-1 V(Very_Dry_39)
+1 V(Very_Dry_40)
= 0

c_e_res_balance(Very_Dry_41)_:
+0.0036 q(Very_Dry_41)
-1 V(Very_Dry_40)
+1 V(Very_Dry_41)
= 0

c_e_res_balance(Very_Dry_42)_:
+0.0036 q(Very_Dry_42)
-1 V(Very_Dry_41)
+1 V(Very_Dry_42)
= 0

c_e_res_balance(Very_Dry_43)_:
+0.0036 q(Very_Dry_43)
-1 V(Very_Dry_42)
+1 V(Very_Dry_43)
= 0

c_e_res_balance(Very_Dry_44)_:
+0.0036 q(Very_Dry_44)
-1 V(Very_Dry_43)
+1 V(Very_Dry_44)
= 0

c_e_res_balance(Very_Dry_45)_:
+0.0036 q(Very_Dry_45)
-1 V(Very_Dry_44)
+1 V(Very_Dry_45)
= 0

c_e_res_balance(Very_Dry_46)_:
+0.0036 q(Very_Dry_46)
-1 V(Very_Dry_45)
+1 V(Very_Dry_46)
= 0

c_e_res_balance(Very_Dry_47)_:
+0.0036 q(Very_Dry_47)
-1 V(Very_Dry_46)
+1 V(Very_Dry_47)
= 0

c_e_res_balance(Very_Dry_48)_:
+0.0036 q(Very_Dry_48)
-1 V(Very_Dry_47)
+1 V(Very_Dry_48)
= 0

c_e_res_balance(Dry_26)_:
+0.0036 q(Dry_26)
-1 V(Dry_25)
+1 V(Dry_26)
= 0.036

c_e_res_balance(Dry_27)_:
+0.0036 q(Dry_27)
-1 V(Dry_26)
+1 V(Dry_27)
= 0.036

c_e_res_balance(Dry_28)_:
+0.0036 q(Dry_28)
-1 V(Dry_27)
+1 V(Dry_28)
= 0.036

c_e_res_balance(Dry_29)_:
+0.0036 q(Dry_29)
-1 V(Dry_28)
+1 V(Dry_29)
= 0.036

c_e_res_balance(Dry_30)_:
+0.0036 q(Dry_30)
-1 V(Dry_29)
+1 V(Dry_30)
= 0.036

c_e_res_balance(Dry_31)_:
+0.0036 q(Dry_31)
-1 V(Dry_30)
+1 V(Dry_31)
= 0.036

c_e_res_balance(Dry_32)_:
+0.0036 q(Dry_32)
-1 V(Dry_31)
+1 V(Dry_32)
= 0.036

c_e_res_balance(Dry_33)_:
+0.0036 q(Dry_33)
-1 V(Dry_32)
+1 V(Dry_33)
= 0.036

c_e_res_balance(Dry_34)_:
+0.0036 q(Dry_34)
-1 V(Dry_33)
+1 V(Dry_34)
= 0.036

c_e_res_balance(Dry_35)_:
+0.0036 q(Dry_35)
-1 V(Dry_34)
+1 V(Dry_35)
= 0.036

c_e_res_balance(Dry_36)_:
+0.0036 q(Dry_36)
-1 V(Dry_35)
+1 V(Dry_36)
= 0.036

c_e_res_balance(Dry_37)_:
+0.0036 q(Dry_37)
-1 V(Dry_36)
+1 V(Dry_37)
= 0.036

c_e_res_balance(Dry_38)_:
+0.0036 q(Dry_38)
-1 V(Dry_37)
+1 V(Dry_38)
= 0.036

c_e_res_balance(Dry_39)_:
+0.0036 q(Dry_39)
-1 V(Dry_38)
+1 V(Dry_39)
= 0.036

c_e_res_balance(Dry_40)_:
+0.0036 q(Dry_40)
-1 V(Dry_39)
+1 V(Dry_40)
= 0.036

c_e_res_balance(Dry_41)_:
+0.0036 q(Dry_41)
-1 V(Dry_40)
+1 V(Dry_41)
= 0.036

c_e_res_balance(Dry_42)_:
+0.0036 q(Dry_42)
-1 V(Dry_41)
+1 V(Dry_42)
= 0.036

c_e_res_balance(Dry_43)_:
+0.0036 q(Dry_43)
-1 V(Dry_42)
+1 V(Dry_43)
= 0.036

c_e_res_balance(Dry_44)_:
+0.0036 q(Dry_44)
-1 V(Dry_43)
+1 V(Dry_44)
= 0.036

c_e_res_balance(Dry_45)_:
+0.0036 q(Dry_45)
-1 V(Dry_44)
+1 V(Dry_45)
= 0.036

c_e_res_balance(Dry_46)_:
+0.0036 q(Dry_46)
-1 V(Dry_45)
+1 V(Dry_46)
= 0.036

c_e_res_balance(Dry_47)_:
+0.0036 q(Dry_47)
-1 V(Dry_46)
+1 V(Dry_47)
= 0.036

c_e_res_balance(Dry_48)_:
+0.0036 q(Dry_48)
-1 V(Dry_47)
+1 V(Dry_48)
= 0.036

c_e_res_balance(Normal_26)_:
+0.0036 q(Normal_26)
-1 V(Normal_25)
+1 V(Normal_26)
= 0.072

c_e_res_balance(Normal_27)_:
+0.0036 q(Normal_27)
-1 V(Normal_26)
+1 V(Normal_27)
= 0.072

c_e_res_balance(Normal_28)_:
+0.0036 q(Normal_28)
-1 V(Normal_27)
+1 V(Normal_28)
= 0.072

c_e_res_balance(Normal_29)_:
+0.0036 q(Normal_29)
-1 V(Normal_28)
+1 V(Normal_29)
= 0.072

c_e_res_balance(Normal_30)_:
+0.0036 q(Normal_30)
-1 V(Normal_29)
+1 V(Normal_30)
= 0.072

c_e_res_balance(Normal_31)_:
+0.0036 q(Normal_31)
-1 V(Normal_30)
+1 V(Normal_31)
= 0.072

c_e_res_balance(Normal_32)_:
+0.0036 q(Normal_32)
-1 V(Normal_31)
+1 V(Normal_32)
= 0.072

c_e_res_balance(Normal_33)_:
+0.0036 q(Normal_33)
-1 V(Normal_32)
+1 V(Normal_33)
= 0.072

c_e_res_balance(Normal_34)_:
+0.0036 q(Normal_34)
-1 V(Normal_33)
+1 V(Normal_34)
= 0.072

c_e_res_balance(Normal_35)_:
+0.0036 q(Normal_35)
-1 V(Normal_34)
+1 V(Normal_35)
= 0.072

c_e_res_balance(Normal_36)_:
+0.0036 q(Normal_36)
-1 V(Normal_35)
+1 V(Normal_36)
= 0.072

c_e_res_balance(Normal_37)_:
+0.0036 q(Normal_37)
-1 V(Normal_36)
+1 V(Normal_37)
= 0.072

c_e_res_balance(Normal_38)_:
+0.0036 q(Normal_38)
-1 V(Normal_37)
+1 V(Normal_38)
= 0.072

c_e_res_balance(Normal_39)_:
+0.0036 q(Normal_39)
-1 V(Normal_38)
+1 V(Normal_39)
= 0.072

c_e_res_balance(Normal_40)_:
+0.0036 q(Normal_40)
-1 V(Normal_39)
+1 V(Normal_40)
= 0.072

c_e_res_balance(Normal_41)_:
+0.0036 q(Normal_41)
-1 V(Normal_40)
+1 V(Normal_41)
= 0.072

c_e_res_balance(Normal_42)_:
+0.0036 q(Normal_42)
-1 V(Normal_41)
+1 V(Normal_42)
= 0.072

c_e_res_balance(Normal_43)_:
+0.0036 q(Normal_43)
-1 V(Normal_42)
+1 V(Normal_43)
= 0.072

c_e_res_balance(Normal_44)_:
+0.0036 q(Normal_44)
-1 V(Normal_43)
+1 V(Normal_44)
= 0.072

c_e_res_balance(Normal_45)_:
+0.0036 q(Normal_45)
-1 V(Normal_44)
+1 V(Normal_45)
= 0.072

c_e_res_balance(Normal_46)_:
+0.0036 q(Normal_46)
-1 V(Normal_45)
+1 V(Normal_46)
= 0.072

c_e_res_balance(Normal_47)_:
+0.0036 q(Normal_47)
-1 V(Normal_46)
+1 V(Normal_47)
= 0.072

c_e_res_balance(Normal_48)_:
+0.0036 q(Normal_48)
-1 V(Normal_47)
+1 V(Normal_48)
= 0.072

c_e_res_balance(Wet_26)_:
+0.0036 q(Wet_26)
-1 V(Wet_25)
+1 V(Wet_26)
= 0.108

c_e_res_balance(Wet_27)_:
+0.0036 q(Wet_27)
-1 V(Wet_26)
+1 V(Wet_27)
= 0.108

c_e_res_balance(Wet_28)_:
+0.0036 q(Wet_28)
-1 V(Wet_27)
+1 V(Wet_28)
= 0.108

c_e_res_balance(Wet_29)_:
+0.0036 q(Wet_29)
-1 V(Wet_28)
+1 V(Wet_29)
= 0.108

c_e_res_balance(Wet_30)_:
+0.0036 q(Wet_30)
-1 V(Wet_29)
+1 V(Wet_30)
= 0.108

c_e_res_balance(Wet_31)_:
+0.0036 q(Wet_31)
-1 V(Wet_30)
+1 V(Wet_31)
= 0.108

c_e_res_balance(Wet_32)_:
+0.0036 q(Wet_32)
-1 V(Wet_31)
+1 V(Wet_32)
= 0.108

c_e_res_balance(Wet_33)_:
+0.0036 q(Wet_33)
-1 V(Wet_32)
+1 V(Wet_33)
= 0.108

c_e_res_balance(Wet_34)_:
+0.0036 q(Wet_34)
-1 V(Wet_33)
+1 V(Wet_34)
= 0.108

c_e_res_balance(Wet_35)_:
+0.0036 q(Wet_35)
-1 V(Wet_34)
+1 V(Wet_35)
= 0.108

c_e_res_balance(Wet_36)_:
+0.0036 q(Wet_36)
-1 V(Wet_35)
+1 V(Wet_36)
= 0.108

c_e_res_balance(Wet_37)_:
+0.0036 q(Wet_37)
-1 V(Wet_36)
+1 V(Wet_37)
= 0.108

c_e_res_balance(Wet_38)_:
+0.0036 q(Wet_38)
-1 V(Wet_37)
+1 V(Wet_38)
= 0.108

c_e_res_balance(Wet_39)_:
+0.0036 q(Wet_39)
-1 V(Wet_38)
+1 V(Wet_39)
= 0.108

c_e_res_balance(Wet_40)_:
+0.0036 q(Wet_40)
-1 V(Wet_39)
+1 V(Wet_40)
= 0.108

c_e_res_balance(Wet_41)_:
+0.0036 q(Wet_41)
-1 V(Wet_40)
+1 V(Wet_41)
= 0.108

c_e_res_balance(Wet_42)_:
+0.0036 q(Wet_42)
-1 V(Wet_41)
+1 V(Wet_42)
= 0.108

c_e_res_balance(Wet_43)_:
+0.0036 q(Wet_43)
-1 V(Wet_42)
+1 V(Wet_43)
= 0.108

c_e_res_balance(Wet_44)_:
+0.0036 q(Wet_44)
-1 V(Wet_43)
+1 V(Wet_44)
= 0.108

c_e_res_balance(Wet_45)_:
+0.0036 q(Wet_45)
-1 V(Wet_44)
+1 V(Wet_45)
= 0.108

c_e_res_balance(Wet_46)_:
+0.0036 q(Wet_46)
-1 V(Wet_45)
+1 V(Wet_46)
= 0.108

c_e_res_balance(Wet_47)_:
+0.0036 q(Wet_47)
-1 V(Wet_46)
+1 V(Wet_47)
= 0.108

c_e_res_balance(Wet_48)_:
+0.0036 q(Wet_48)
-1 V(Wet_47)
+1 V(Wet_48)
= 0.108

c_e_res_balance(Very_Wet_26)_:
+0.0036 q(Very_Wet_26)
-1 V(Very_Wet_25)
+1 V(Very_Wet_26)
= 0.144

c_e_res_balance(Very_Wet_27)_:
+0.0036 q(Very_Wet_27)
-1 V(Very_Wet_26)
+1 V(Very_Wet_27)
= 0.144

c_e_res_balance(Very_Wet_28)_:
+0.0036 q(Very_Wet_28)
-1 V(Very_Wet_27)
+1 V(Very_Wet_28)
= 0.144

c_e_res_balance(Very_Wet_29)_:
+0.0036 q(Very_Wet_29)
-1 V(Very_Wet_28)
+1 V(Very_Wet_29)
= 0.144

c_e_res_balance(Very_Wet_30)_:
+0.0036 q(Very_Wet_30)
-1 V(Very_Wet_29)
+1 V(Very_Wet_30)
= 0.144

c_e_res_balance(Very_Wet_31)_:
+0.0036 q(Very_Wet_31)
-1 V(Very_Wet_30)
+1 V(Very_Wet_31)
= 0.144

c_e_res_balance(Very_Wet_32)_:
+0.0036 q(Very_Wet_32)
-1 V(Very_Wet_31)
+1 V(Very_Wet_32)
= 0.144

c_e_res_balance(Very_Wet_33)_:
+0.0036 q(Very_Wet_33)
-1 V(Very_Wet_32)
+1 V(Very_Wet_33)
= 0.144

c_e_res_balance(Very_Wet_34)_:
+0.0036 q(Very_Wet_34)
-1 V(Very_Wet_33)
+1 V(Very_Wet_34)
= 0.144

c_e_res_balance(Very_Wet_35)_:
+0.0036 q(Very_Wet_35)
-1 V(Very_Wet_34)
+1 V(Very_Wet_35)
= 0.144

c_e_res_balance(Very_Wet_36)_:
+0.0036 q(Very_Wet_36)
-1 V(Very_Wet_35)
+1 V(Very_Wet_36)
= 0.144

c_e_res_balance(Very_Wet_37)_:
+0.0036 q(Very_Wet_37)
-1 V(Very_Wet_36)
+1 V(Very_Wet_37)
= 0.144

c_e_res_balance(Very_Wet_38)_:
+0.0036 q(Very_Wet_38)
-1 V(Very_Wet_37)
+1 V(Very_Wet_38)
= 0.144

c_e_res_balance(Very_Wet_39)_:
+0.0036 q(Very_Wet_39)
-1 V(Very_Wet_38)
+1 V(Very_Wet_39)
= 0.144

c_e_res_balance(Very_Wet_40)_:
+0.0036 q(Very_Wet_40)
-1 V(Very_Wet_39)
+1 V(Very_Wet_40)
= 0.144

c_e_res_balance(Very_Wet_41)_:
+0.0036 q(Very_Wet_41)
-1 V(Very_Wet_40)
+1 V(Very_Wet_41)
= 0.144

c_e_res_balance(Very_Wet_42)_:
+0.0036 q(Very_Wet_42)
-1 V(Very_Wet_41)
+1 V(Very_Wet_42)
= 0.144

c_e_res_balance(Very_Wet_43)_:
+0.0036 q(Very_Wet_43)
-1 V(Very_Wet_42)
+1 V(Very_Wet_43)
= 0.144

c_e_res_balance(Very_Wet_44)_:
+0.0036 q(Very_Wet_44)
-1 V(Very_Wet_43)
+1 V(Very_Wet_44)
= 0.144

c_e_res_balance(Very_Wet_45)_:
+0.0036 q(Very_Wet_45)
-1 V(Very_Wet_44)
+1 V(Very_Wet_45)
= 0.144

c_e_res_balance(Very_Wet_46)_:
+0.0036 q(Very_Wet_46)
-1 V(Very_Wet_45)
+1 V(Very_Wet_46)
= 0.144

c_e_res_balance(Very_Wet_47)_:
+0.0036 q(Very_Wet_47)
-1 V(Very_Wet_46)
+1 V(Very_Wet_47)
= 0.144

c_e_res_balance(Very_Wet_48)_:
+0.0036 q(Very_Wet_48)
-1 V(Very_Wet_47)
+1 V(Very_Wet_48)
= 0.144

bounds
   0 <= q(Very_Dry_25) <= 36.571960087941825
   0 <= q(Very_Dry_26) <= 36.571960087941825
   0 <= q(Very_Dry_27) <= 36.571960087941825
   0 <= q(Very_Dry_28) <= 36.571960087941825
   0 <= q(Very_Dry_29) <= 36.571960087941825
   0 <= q(Very_Dry_30) <= 36.571960087941825
   0 <= q(Very_Dry_31) <= 36.571960087941825
   0 <= q(Very_Dry_32) <= 36.571960087941825
   0 <= q(Very_Dry_33) <= 36.571960087941825
   0 <= q(Very_Dry_34) <= 36.571960087941825
   0 <= q(Very_Dry_35) <= 36.571960087941825
   0 <= q(Very_Dry_36) <= 36.571960087941825
   0 <= q(Very_Dry_37) <= 36.571960087941825
   0 <= q(Very_Dry_38) <= 36.571960087941825
   0 <= q(Very_Dry_39) <= 36.571960087941825
   0 <= q(Very_Dry_40) <= 36.571960087941825
   0 <= q(Very_Dry_41) <= 36.571960087941825
   0 <= q(Very_Dry_42) <= 36.571960087941825
   0 <= q(Very_Dry_43) <= 36.571960087941825
   0 <= q(Very_Dry_44) <= 36.571960087941825
   0 <= q(Very_Dry_45) <= 36.571960087941825
   0 <= q(Very_Dry_46) <= 36.571960087941825
   0 <= q(Very_Dry_47) <= 36.571960087941825
   0 <= q(Very_Dry_48) <= 36.571960087941825
   0 <= q(Dry_25) <= 36.571960087941825
   0 <= q(Dry_26) <= 36.571960087941825
   0 <= q(Dry_27) <= 36.571960087941825
   0 <= q(Dry_28) <= 36.571960087941825
   0 <= q(Dry_29) <= 36.571960087941825
   0 <= q(Dry_30) <= 36.571960087941825
   0 <= q(Dry_31) <= 36.571960087941825
   0 <= q(Dry_32) <= 36.571960087941825
   0 <= q(Dry_33) <= 36.571960087941825
   0 <= q(Dry_34) <= 36.571960087941825
   0 <= q(Dry_35) <= 36.571960087941825
   0 <= q(Dry_36) <= 36.571960087941825
   0 <= q(Dry_37) <= 36.571960087941825
   0 <= q(Dry_38) <= 36.571960087941825
   0 <= q(Dry_39) <= 36.571960087941825
   0 <= q(Dry_40) <= 36.571960087941825
   0 <= q(Dry_41) <= 36.571960087941825
   0 <= q(Dry_42) <= 36.571960087941825
   0 <= q(Dry_43) <= 36.571960087941825
   0 <= q(Dry_44) <= 36.571960087941825
   0 <= q(Dry_45) <= 36.571960087941825
   0 <= q(Dry_46) <= 36.571960087941825
   0 <= q(Dry_47) <= 36.571960087941825
   0 <= q(Dry_48) <= 36.571960087941825
   0 <= q(Normal_25) <= 36.571960087941825
   0 <= q(Normal_26) <= 36.571960087941825
   0 <= q(Normal_27) <= 36.571960087941825
   0 <= q(Normal_28) <= 36.571960087941825
   0 <= q(Normal_29) <= 36.571960087941825
   0 <= q(Normal_30) <= 36.571960087941825
   0 <= q(Normal_31) <= 36.571960087941825
   0 <= q(Normal_32) <= 36.571960087941825
   0 <= q(Normal_33) <= 36.571960087941825
   0 <= q(Normal_34) <= 36.571960087941825
   0 <= q(Normal_35) <= 36.571960087941825
   0 <= q(Normal_36) <= 36.571960087941825
   0 <= q(Normal_37) <= 36.571960087941825
   0 <= q(Normal_38) <= 36.571960087941825
   0 <= q(Normal_39) <= 36.571960087941825
   0 <= q(Normal_40) <= 36.571960087941825
   0 <= q(Normal_41) <= 36.571960087941825
   0 <= q(Normal_42) <= 36.571960087941825
   0 <= q(Normal_43) <= 36.571960087941825
   0 <= q(Normal_44) <= 36.571960087941825
   0 <= q(Normal_45) <= 36.571960087941825
   0 <= q(Normal_46) <= 36.571960087941825
   0 <= q(Normal_47) <= 36.571960087941825
   0 <= q(Normal_48) <= 36.571960087941825
   0 <= q(Wet_25) <= 36.571960087941825
   0 <= q(Wet_26) <= 36.571960087941825
   0 <= q(Wet_27) <= 36.571960087941825
   0 <= q(Wet_28) <= 36.571960087941825
   0 <= q(Wet_29) <= 36.571960087941825
   0 <= q(Wet_30) <= 36.571960087941825
   0 <= q(Wet_31) <= 36.571960087941825
   0 <= q(Wet_32) <= 36.571960087941825
   0 <= q(Wet_33) <= 36.571960087941825
   0 <= q(Wet_34) <= 36.571960087941825
   0 <= q(Wet_35) <= 36.571960087941825
   0 <= q(Wet_36) <= 36.571960087941825
   0 <= q(Wet_37) <= 36.571960087941825
   0 <= q(Wet_38) <= 36.571960087941825
   0 <= q(Wet_39) <= 36.571960087941825
   0 <= q(Wet_40) <= 36.571960087941825
   0 <= q(Wet_41) <= 36.571960087941825
   0 <= q(Wet_42) <= 36.571960087941825
   0 <= q(Wet_43) <= 36.571960087941825
   0 <= q(Wet_44) <= 36.571960087941825
   0 <= q(Wet_45) <= 36.571960087941825
   0 <= q(Wet_46) <= 36.571960087941825
   0 <= q(Wet_47) <= 36.571960087941825
   0 <= q(Wet_48) <= 36.571960087941825
   0 <= q(Very_Wet_25) <= 36.571960087941825
   0 <= q(Very_Wet_26) <= 36.571960087941825
   0 <= q(Very_Wet_27) <= 36.571960087941825
   0 <= q(Very_Wet_28) <= 36.571960087941825
   0 <= q(Very_Wet_29) <= 36.571960087941825
   0 <= q(Very_Wet_30) <= 36.571960087941825
   0 <= q(Very_Wet_31) <= 36.571960087941825
   0 <= q(Very_Wet_32) <= 36.571960087941825
   0 <= q(Very_Wet_33) <= 36.571960087941825
   0 <= q(Very_Wet_34) <= 36.571960087941825
   0 <= q(Very_Wet_35) <= 36.571960087941825
   0 <= q(Very_Wet_36) <= 36.571960087941825
   0 <= q(Very_Wet_37) <= 36.571960087941825
   0 <= q(Very_Wet_38) <= 36.571960087941825
   0 <= q(Very_Wet_39) <= 36.571960087941825
   0 <= q(Very_Wet_40) <= 36.571960087941825
   0 <= q(Very_Wet_41) <= 36.571960087941825
   0 <= q(Very_Wet_42) <= 36.571960087941825
   0 <= q(Very_Wet_43) <= 36.571960087941825
   0 <= q(Very_Wet_44) <= 36.571960087941825
   0 <= q(Very_Wet_45) <= 36.571960087941825
   0 <= q(Very_Wet_46) <= 36.571960087941825
   0 <= q(Very_Wet_47) <= 36.571960087941825
   0 <= q(Very_Wet_48) <= 36.571960087941825
   0 <= V(Very_Dry_25) <= 4.5
   0 <= V(Very_Dry_26) <= 4.5
   0 <= V(Very_Dry_27) <= 4.5
   0 <= V(Very_Dry_28) <= 4.5
   0 <= V(Very_Dry_29) <= 4.5
   0 <= V(Very_Dry_30) <= 4.5
   0 <= V(Very_Dry_31) <= 4.5
   0 <= V(Very_Dry_32) <= 4.5
   0 <= V(Very_Dry_33) <= 4.5
   0 <= V(Very_Dry_34) <= 4.5
   0 <= V(Very_Dry_35) <= 4.5
   0 <= V(Very_Dry_36) <= 4.5
   0 <= V(Very_Dry_37) <= 4.5
   0 <= V(Very_Dry_38) <= 4.5
   0 <= V(Very_Dry_39) <= 4.5
   0 <= V(Very_Dry_40) <= 4.5
   0 <= V(Very_Dry_41) <= 4.5
   0 <= V(Very_Dry_42) <= 4.5
   0 <= V(Very_Dry_43) <= 4.5
   0 <= V(Very_Dry_44) <= 4.5
   0 <= V(Very_Dry_45) <= 4.5
   0 <= V(Very_Dry_46) <= 4.5
   0 <= V(Very_Dry_47) <= 4.5
   0 <= V(Very_Dry_48) <= 4.5
   0 <= V(Dry_25) <= 4.5
   0 <= V(Dry_26) <= 4.5
   0 <= V(Dry_27) <= 4.5
   0 <= V(Dry_28) <= 4.5
   0 <= V(Dry_29) <= 4.5
   0 <= V(Dry_30) <= 4.5
   0 <= V(Dry_31) <= 4.5
   0 <= V(Dry_32) <= 4.5
   0 <= V(Dry_33) <= 4.5
   0 <= V(Dry_34) <= 4.5
   0 <= V(Dry_35) <= 4.5
   0 <= V(Dry_36) <= 4.5
   0 <= V(Dry_37) <= 4.5
   0 <= V(Dry_38) <= 4.5
   0 <= V(Dry_39) <= 4.5
   0 <= V(Dry_40) <= 4.5
   0 <= V(Dry_41) <= 4.5
   0 <= V(Dry_42) <= 4.5
   0 <= V(Dry_43) <= 4.5
   0 <= V(Dry_44) <= 4.5
   0 <= V(Dry_45) <= 4.5
   0 <= V(Dry_46) <= 4.5
   0 <= V(Dry_47) <= 4.5
   0 <= V(Dry_48) <= 4.5
   0 <= V(Normal_25) <= 4.5
   0 <= V(Normal_26) <= 4.5
   0 <= V(Normal_27) <= 4.5
   0 <= V(Normal_28) <= 4.5
   0 <= V(Normal_29) <= 4.5
   0 <= V(Normal_30) <= 4.5
   0 <= V(Normal_31) <= 4.5
   0 <= V(Normal_32) <= 4.5
   0 <= V(Normal_33) <= 4.5
   0 <= V(Normal_34) <= 4.5
   0 <= V(Normal_35) <= 4.5
   0 <= V(Normal_36) <= 4.5
   0 <= V(Normal_37) <= 4.5
   0 <= V(Normal_38) <= 4.5
   0 <= V(Normal_39) <= 4.5
   0 <= V(Normal_40) <= 4.5
   0 <= V(Normal_41) <= 4.5
   0 <= V(Normal_42) <= 4.5
   0 <= V(Normal_43) <= 4.5
   0 <= V(Normal_44) <= 4.5
   0 <= V(Normal_45) <= 4.5
   0 <= V(Normal_46) <= 4.5
   0 <= V(Normal_47) <= 4.5
   0 <= V(Normal_48) <= 4.5
   0 <= V(Wet_25) <= 4.5
   0 <= V(Wet_26) <= 4.5
   0 <= V(Wet_27) <= 4.5
   0 <= V(Wet_28) <= 4.5
   0 <= V(Wet_29) <= 4.5
   0 <= V(Wet_30) <= 4.5
   0 <= V(Wet_31) <= 4.5
   0 <= V(Wet_32) <= 4.5
   0 <= V(Wet_33) <= 4.5
   0 <= V(Wet_34) <= 4.5
   0 <= V(Wet_35) <= 4.5
   0 <= V(Wet_36) <= 4.5
   0 <= V(Wet_37) <= 4.5
   0 <= V(Wet_38) <= 4.5
   0 <= V(Wet_39) <= 4.5
   0 <= V(Wet_40) <= 4.5
   0 <= V(Wet_41) <= 4.5
   0 <= V(Wet_42) <= 4.5
   0 <= V(Wet_43) <= 4.5
   0 <= V(Wet_44) <= 4.5
   0 <= V(Wet_45) <= 4.5
   0 <= V(Wet_46) <= 4.5
   0 <= V(Wet_47) <= 4.5
   0 <= V(Wet_48) <= 4.5
   0 <= V(Very_Wet_25) <= 4.5
   0 <= V(Very_Wet_26) <= 4.5
   0 <= V(Very_Wet_27) <= 4.5
   0 <= V(Very_Wet_28) <= 4.5
   0 <= V(Very_Wet_29) <= 4.5
   0 <= V(Very_Wet_30) <= 4.5
   0 <= V(Very_Wet_31) <= 4.5
   0 <= V(Very_Wet_32) <= 4.5
   0 <= V(Very_Wet_33) <= 4.5
   0 <= V(Very_Wet_34) <= 4.5
   0 <= V(Very_Wet_35) <= 4.5
   0 <= V(Very_Wet_36) <= 4.5
   0 <= V(Very_Wet_37) <= 4.5
   0 <= V(Very_Wet_38) <= 4.5
   0 <= V(Very_Wet_39) <= 4.5
   0 <= V(Very_Wet_40) <= 4.5
   0 <= V(Very_Wet_41) <= 4.5
   0 <= V(Very_Wet_42) <= 4.5
   0 <= V(Very_Wet_43) <= 4.5
   0 <= V(Very_Wet_44) <= 4.5
   0 <= V(Very_Wet_45) <= 4.5
   0 <= V(Very_Wet_46) <= 4.5
   0 <= V(Very_Wet_47) <= 4.5
   0 <= V(Very_Wet_48) <= 4.5
end
