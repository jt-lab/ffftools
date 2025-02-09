# fff-tools

Python module for preprocessing data from visual foraging experiments that was stored in fff ("foraging file format").
The fff defines a set of mandatory (and optional supplementary) columns that foraging experiments need to store. Based on these, fff can produce a set computable columns. For instance, based on the mandatory column M_Selection_Time, fff tools can produce C_Selection_Inter-target_Time. Based on M_Selection_Time and C_Selection_Value, fff tools can compute C_Selection_Rate_of_Return (see subfolder "columns" for all currently covered variables). The fff also defines default units, coordinate sytems, etc. and fff tools provide commonly used plots. Everything is still heavily under construction.

