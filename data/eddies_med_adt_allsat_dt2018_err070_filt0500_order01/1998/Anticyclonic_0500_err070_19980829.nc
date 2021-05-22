CDF       
      obs    9   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?�M����      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M�*D   max       P��      �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       �@�   max       >1'      �  t   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?�z�H   max       @E�(�\     �   X   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?��Q�    max       @v��Q�     �  )@   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @%         max       @P            t  2(   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�^        max       @���          �  2�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ���   max       >ix�      �  3�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�C(   max       B/;�      �  4d   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�|#   max       B/CD      �  5H   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       ?Q   max       C��z      �  6,   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       ?Y��   max       C��k      �  7   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          �      �  7�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min          
   max          I      �  8�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min          
   max          3      �  9�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M�*D   max       P�5      �  :�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�s�g��   max       ?�jOv`      �  ;�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       �'�   max       >1'      �  <h   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?�z�H   max       @E�(�\     �  =L   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?��Q�    max       @v�p��
>     �  F4   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @#         max       @N�           t  O   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�^        max       @��          �  O�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         El   max         El      �  Pt   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?���S���   max       ?�jOv`     �  QX                  3   J               5            $      X      	                  
         6   B      9   t   M   �            0      &          )      1   0      I      	   /   �      	   7   O#pN��N9��M��&NE�P8��P��O�˧N �O+mN!3�O��N!S{N��N�fP
�O�yRP�iN�.N��N��Oϻ'O)�N;�N��N�O�8�M�*DP��XO�3�O ��PD�lPUPd�~P7u�O8+bO��N�$_OUqO�6O��=O��_O%!O�Ny�O��sO�gN�H�O�L�N&uN���O�CCO�qbN6�NO��Oe�yNl5z�@��#�
���ͼ�o�e`B�49X��o;��
;��
;ě�;�`B<#�
<#�
<D��<T��<T��<�o<�C�<�C�<�t�<�t�<���<��
<�9X<ě�<���<�h=C�=\)=�w=�w=#�
='�='�=,1=<j=@�=@�=@�=P�`=T��=]/=]/=aG�=q��=�%=�%=��=��=���=��T=�{=� �=��=�
==��>1'��������������������845<AHMIH<8888888888b]gtxyxtpgbbbbbbbbbb����������������������������������������������
*9GCDB3#�������
/HY\QUavynG<����������������������������������������������������������������

 ���������#<HMT]`^XQKH</*!��������������������&)55<?:55)!���������� ���������
/;H[afdaH</#RTVamz��������zmfa]R����5W[YHD>5���().36@BNOPOKICB764)(46<9BEOP[\[[WOGB:644*,-+*��������� ������������������������������!"+/18:/"�����������������������������������������������������������:5:<IMII@<::::::::::�����)BGLMI5����������������������������������������������������������������kkr��������������zpk����������������	)BOY^htlqnpyB)		#/1<ADC<<;/+#	-/08<HNUWZ\`UH<5//--hdht�������������xth��������������������%16BGOT[aO@6)	)BTY\VOB)�������(')%!����������������������������$(&����������������������_beem�����������zmc_MQ[gt���������tg[TOM�����������������������)IXVOHF@6)����������������������
#).%#!
MGEBNO[gt�������tg[M���������

�������	����������������

���������

������
"#)'#
���
������
������������������������D�D�D�ED�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�������������������������������������²¿��������¿²¯²²²²²²²²²²²������������������������������������������(�F�N�Z�p�{�}�g�Z�A�5����������	�"�a�}�����������m�T�;�"�������������	���(�5�A�Q�Z�c�e�a�Z�N�5�(��	���������	���������������������ûлֻۻڻл̻û������������������������������þǾ������������������������������$�"�����������ù÷ü����������ÇÓßàààÕÓÇÂÇÊÇÅÇÇÇÇÇÇ�����þʾ׾پ׾׾ʾ��������������������������	���"�%�"����	���������m����z�m�`�G�;�"�	��������"�;�G�m����	����	��������������������������Ƴ���������������ƚ�u�\�O�6���OƇƳ�"�%�/�;�@�H�J�T�W�H�A�;�/�"� �����"���������������������~�z�m�f�m�p�z�������y�������������������y�x�m�k�m�o�y�y�y�y�� �@�Y�����������x�f�Y�4������ ��"�/�;�H�R�T�_�\�T�M�H�;�/�*�"����"�"�a�m�q�z���z�m�a�\�_�a�a�a�a�a�a�a�a�a�a������ �!�!�!���������������Ŀѿݿ����������ݿѿǿĿ��������ľ4�A�M�Z�f�s�}�~�y�s�f�Z�A���
���(�4�r�����������}�r�q�r�r�r�r�r�r�r�r�r�r�(�:�E�E�P�]�\�c�W�N�5����ɿ˿׿���(�������/��������ܻû����������л������������������������������}�~���������������Ľ����A�R�R�D�=�(���齫�������h�tčĦį������Ŀĳď�t�S�B�8�0�6�<�O�h�����'�!�/�/�>�6�)����ýìÏÌà������Y�r�����������ʼҼ����r�M�@�4�(�+�2�@�Y�����������������s�f�M�I�M�V�Z�_�f�s���&�(�4�=�<�4�,�(���������������<�H�I�I�O�S�M�H�<�3�/�.�$�#�!�#�&�/�5�<ÇÓàìóùþ��üìàÓÇ�}�z�w�x�zÀÇ�r�~���������ɺպɺ����������r�^�M�V�e�r�ʾ׾���	�"�'�"������׾ʾ��������������
��#�0�<�B�O�T�T�I�<�0�#��
��������F$F1F=FVFcFdFqFvFzFrFoFcFaFJF=F1F*F$FF$������������������y�l�`�S�L�G�E�H�S�l������������������������������������������ŭ������������������������ŷŮŪśŗŠŭ���*�-�-�)�"�����������������������)�+�6�;�B�P�S�O�B�=�8�6�1�)�'�!�"���!�:�S�l�r�o�o�_�S�:�/�!�����������!�[�g�t�t�t�l�g�[�Y�O�[�[�[�[�[�[�[�[�[�[�[�g�n�g�e�^�[�N�B�A�B�G�N�T�[�[�[�[�[�[�����	��!�"�+�-�*�$�"��	��������������D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�DvDoDsD{D�ǭǭǡǔǈǊǔǡǦǭǭǭǭǭǭǭǭǭǭǭ�I�U�b�d�n�t�n�b�U�I�I�C�I�I�I�I�I�I�I�IEuE�E�E�E�E�E�E�E�E�E�E�E�E�EzEtEiEeEiEu�C�O�T�[�O�M�C�6�4�2�6�:�C�C�C�C�C�C�C�C O J f _ W * 5 * i , { : _ 8 P K K F p @ - | - I 4 \ = K 4 = ( F Q ; < > ) Y # U Q A f p Q P  � $ 1 n *  ' T 4 0    p  )  F      %  �  u  f  @  h  n  S  �     �  �  �  �  �  �  t  m  R  �  E  �    j  �    �    �  ;  �  1  A  �  �      �  ^  �  �  �  8  &  B  �  
  X  R  }  �  o��/��㼴9X�o�t�=#�
=�O�=t�<t�<�9X<49X=}�<�o<���<���=H�9=+=�
=<��
<�/<�1=8Q�=\)<�`B=\)=\)=]/=��=�E�=�
==T��=Ƨ�>�w=��>;dZ=�t�=�%=�\)=��=��=�Q�=�{=���=\=�\)=�S�=�G�=�{>I�=�^5=�E�>$�>ix�=�S�=�x�>$�/>�uB�B�B	�tB ibBB�B_aB��B A�B#�B#ڝBǮBIVB��B��BmQA���B(�B�B�[B/;�B!��B@�A�C(B �!B
͛B�AB&k�B�B�DB��B!�B%�BR�B��B�8B'EB�B!�nB��B�#BsBCzB-1=Bf�B ��B	��B� B#BO�BI�B	��B�B(�B��B�zBN!B��B��B	FuB ?�B�6B�/BnvBŽB m�B#=QB#��B�B?�B�KB�"B��A�l�B�sB��B��B/CDB!�oBB�A�|#B @�B
�zB��B&DRB?�B�2BA�B!��B5�B�GB�B� B7�BZYB"6�B��B��B@�BQB-��B��B_}B	C�B�B�<BAkB��B	��B�,B:�B�CB��B?�A��C�8�A��OA���A��,A��wA��DA�M5?Q@�ALc�A҈]A�<�AN�)AZAcI�A��JB >A��A�!UAnΆ@���A�ZoA�_�@]��A}#"A;�u@�A�W1@�3�A���A-7�A��AѦ�@޽�AD�A4�NA�gOA��@)%AU��A�=�C��zA,=A�OA�6_A��DA��!@x�lA��A�:�A�]dC���B��A���C��B �A��C�<�A�'�A�5A�r�A��IA��FA�;?Y��@���AKUAҊzA��AO!�AY�Ab�TA��B��A�w�A��lAnɛ@��WA�~�A�h�@\O5A}�A93@��A�|@��A��A-KA�}�A�}z@�
�AD�A5�A�~�Aʬv@��AR��A��qC��kA�A��A�w�A�^�A�U�@x"A�h�A��KA�y�C�؈BN�A�w�C�ZB �'                  4   J                5            %      Y      
                  
         7   C      :   t   N   �            0      '          )      1   0      I      	   /   �      
   8                     -   I                           '   #   ?            +                     3   %      2   +   1   /               #   #         #      !         #               
                           %                              '   #   3                                 1         2      +                  #   #                        #               
         N�0N��N9��M��&NE�O��HO��ON �O+mN!3�N��N!S{N��N�fO���O�yRP�5N�.Ni<�N��OA��OtiN;�NGdjN�O�8�M�*DPt6O��&O ��PD�lO�?�P4"�O�f�O&$�O��N�W�O:E6O�6O��OE\bN�$O�d�Ny�O�	2OWF2N�BOO��LN&uN���OV�OZ,�N6�NO��OhcNl5z  �  �  {  [    �    �  �  u  %  �  n  �  �  �  �  �      �  �  =  C  e  �  >  _  �  	�  J  g  �  	X  ^  �  y  �  
$  Y  e  �  `  �  �  	�  	�  /  	�  $  �  	�  B  '  �    �'#�
���ͼ�o�e`B%   =t�<D��;��
;ě�;�`B=��<#�
<D��<T��<�o<�o=o<�C�<���<�t�<�h<�9X<�9X<���<���<�h=C�=#�
=T��=�w=#�
=�\)=P�`=���=D��=@�=L��=T��=P�`=Y�=u=m�h=q��=q��=�C�=�O�=�+=�\)=���=��T=�Q�=���=��=�
==��m>1'��������������������845<AHMIH<8888888888b]gtxyxtpgbbbbbbbbbb����������������������������������������������#/5:;6/#���������
#/67:BB8/#
 ���������������������������������������������������������������

 ���������,*,/2<AHLQNH@<7/,,,,��������������������&)55<?:55)!���������� ���������
/8FY_dcaUH</RTVamz��������zmfa]R���)5DNRQC=5)��().36@BNOPOKICB764)(;6@BGOZZTOJB;;;;;;;;*,-+*����������������������������������������!"+/18:/"�����������������������������������������������������������:5:<IMII@<::::::::::�����5<GIB3������������������������������������������������������������������utuy�������������zu�������� �������)6BOW\]YOB6)#/;<@CA></-#-/08<HNUWZ\`UH<5//--ptx�����������ztpppp��������������������%16BGOT[aO@6)	BSX\UOB6)�������$!������������������������������%#�����������������������kedfmv�����������zmkVRPT[cgt��������tg[V�������������������� ��)BNQRKFD<6) ���������������������
#).%#!
UPOOU[gt��������tg[U���������
��������	����������������

�������� ��������
"#)'#
�������
���
��������������������������D�D�D�ED�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�������������������������������������²¿��������¿²¯²²²²²²²²²²²�����������������������������������������(�5�A�K�Z�`�g�`�W�N�A�����������(��/�;�T�a�k�n�o�e�T�H�;�"��	����	���(�A�N�\�_�\�Z�R�N�A�(���	��	��������	���������������������ûлֻۻڻл̻û������������������������������þǾ�������������������������������������	�����������������������ÇÓßàààÕÓÇÂÇÊÇÅÇÇÇÇÇÇ�����þʾ׾پ׾׾ʾ��������������������������	���"�%�"����	���������`�m�|���w�m�`�G�;�"��	���������"�;�`����	����	��������������������������Ƴ�������	���������ƚ�u�^�R�L�J�OƠƳ�"�%�/�;�@�H�J�T�W�H�A�;�/�"� �����"�z�������������z�m�i�m�q�z�z�z�z�z�z�z�z�y�������������������y�x�m�k�m�o�y�y�y�y���4�M�c�r���r�f�Y�@�4�'��������/�;�H�O�T�Z�X�T�H�H�;�0�/�#�"���"�&�/�a�m�q�z���z�m�a�\�_�a�a�a�a�a�a�a�a�a�a���������������������������������Ŀѿݿ����������ݿѿǿĿ��������ľ4�A�M�Z�f�s�}�~�y�s�f�Z�A���
���(�4�r�����������}�r�q�r�r�r�r�r�r�r�r�r�r�(�5�@�@�K�W�S�N�5�����ӿοֿֿܿ���(������������ܻлû����������û�������������������������������}�~���������������Ľ����A�R�R�D�=�(���齫�������h�tāčĚĦıĺĿĿĳĦĚā�t�f�Q�O�Z�h�������&�/�4�)���������ìÛÛ������M�Y�f�r�������������r�f�Y�L�A�<�>�H�M�s�����������������s�f�Z�S�X�Z�`�f�h�s��&�(�4�=�<�4�,�(���������������<�A�H�L�Q�K�H�<�/�&�#�"�#�)�/�9�<�<�<�<ÇÓàíùü��úìàÓÇÀ�z�y�z�z�~ÃÇ�r�~���������ɺպɺ����������r�^�M�V�e�r�ʾ׾���	��"�"�����׾ʾ����������������
��#�0�9�<�G�I�I�C�<�0�#��
��������F$F1F=FJFVF`FcFmFrFoFcFVFJFAF=F1F,F$F"F$�l�y���������������y�l�`�S�I�G�J�S�`�i�l����������������������������������������ŠŭŹ����������������������źŰŭşŞŠ������&�)�*�+�'��������������������)�6�;�B�D�O�O�R�O�B�>�9�6�.�)�(�!�#�)�)�!�:�S�h�o�k�^�S�F�:�-�!������������!�[�g�t�t�t�l�g�[�Y�O�[�[�[�[�[�[�[�[�[�[�[�g�n�g�e�^�[�N�B�A�B�G�N�T�[�[�[�[�[�[�����	��"�&�*�(�!���	����������������D�D�D�D�D�D�D�D�D�D�D�D�D�D�DDxD{D~D�D�ǭǭǡǔǈǊǔǡǦǭǭǭǭǭǭǭǭǭǭǭ�I�U�b�d�n�t�n�b�U�I�I�C�I�I�I�I�I�I�I�IE�E�E�E�E�E�E�E�E�E�E�E�E�EzEuEtEuE|E�E��C�O�T�[�O�M�C�6�4�2�6�:�C�C�C�C�C�C�C�C M J f _ W $  0 i , {  _ 8 P P K 8 p 6 - z - I < \ = K 6 3 ( F G 3  1 ) B " U O : ^ g Q Q  t $ 1 n !  ' T * 0    �  )  F      ;  �    f  @  h  �  S  �     f  �  G  �  p  �  5  8  R  ^  E  �    �  o    �  �      j  1  �  �  �    �  6  �  �  �  �  �  �  B  �  �  �  R  }  %  o  El  El  El  El  El  El  El  El  El  El  El  El  El  El  El  El  El  El  El  El  El  El  El  El  El  El  El  El  El  El  El  El  El  El  El  El  El  El  El  El  El  El  El  El  El  El  El  El  El  El  El  El  El  El  El  El  El  l  �  �  �  �  �  �  �  �  �  �  �  v  _  <  +  �  �  i    �  �  �  �  �  �  �  �  �  �  ~  t  j  a  W  N  D  :  1  '  {  w  s  n  j  f  a  _  ]  [  Y  X  V  K  /    �  �  �  �  [  �    /  3  3  1  0  .  +  (  $  !            �  �      �  �  �  �  �  �  �  �      '  3  >  J  V  b  n  z  �  *  ]  {  �  �  ~  y  o  ^  E  +    �  �  �  3  �    H  �    [  w  �  �  �  �  �      	  �  �  �  ^    �  �  {  �  �  �  �  �  �  �  �  �  o  G     �  �  �  V  �  �    z  �  �  �  �  �  �  �  }  j  X  D  /      �  �  �  �  �  �  u  n  d  Z  O  B  4  $    �  �  �  �  c  0  �  �  �  �  �  %          �  �  �  �  �  �  �  �  �  �  �  �  �  �  �     :  ]  x  �  �  �  �  �  �  �  �  �  m    �    I  >  �  n  g  _  X  P  H  A  C  J  Q  5  �  �  r  ,  �  �  Q    �  �  }  s  h  \  P  C  6  (    	  �  �  �  �  i  >     �   �  �  �  �  �  �  �  �  �  �  z  k  [  J  :  )      ;  d  �  �  �  �  �  �  �  �  �  t  ?    �  {  /  �  �  4  �  t  8  �  �  �  �  �  �  �  �  k  R  8      �  �  �  C  �  �  (  �  [  �  �  �  �  �  ^  B    �  �  [  +  �  �    Q  �  �        �  �  �  �  �  �  �  �  �  �  �  �  �  �    u  k  �  �    %  6  ;  <  ;  9  5  0  ,  +       �  �  �  p  G  �  �  �  }  p  c  W  F  3  !     �   �   �   �   �   �   �   �   �  ,  3  S  {  �  �  �  �  �  �  �  �  g  A      �  �    �  6  9  <  =  <  7  (    �  �  �  �  �  i  >    �  e  	   �  C  =  6  0  &        �  �  �  �  �  �  l  O  0    �  �  ^  _  `  d  c  ^  R  E  =  4  (  	  �  �  �  f  ?    �  �  �  �  �  �  �  �  �  �  �  u  b  K  -    �  �  B  �  �  m  >  9  5  /  $    �  �  �  �  �  �  �  �  �  5  �  I  �  ;  _  Y  S  M  G  ?  3  '         �  �  �  �  �  �  �  �  r  �  �  �  �  �  �  m  V  C  1    �  �  j    �  y  �  )   �  	X  	s  	�  	�  	�  	�  	�  	b  	%  �  �  4  �  W  �    >  T  �   �  J  ?  0        �  �  �  �  �  �  |  h  S  <  "    �  �  g  U  H  S  _  [  U  O  P  U  U  I  4    �  f  �  D  �  w  V    O  h  y    y  n  M    �  �  
  m  �  �  
~  �  4  �  	  	1  	P  	V  	:  	  �  �  �  f    �  h    �  �  d  �  "   �  	�  
�  K    �  �  1  T  ]  J    �  P  �  �  
�  	�  A  $    y  �  �  �  t  R  ,    �  �  b  $  �  �  5  �  �  F  �  2  y  w  t  p  k  c  U  C  (    �  �  q  :              j  �  �  �  �  �  �  y  R  $  �  �  n  '  �  �  v  B  	  c  
  
  
#  
  
  	�  	�  	z  	/  �  s    �    �    �  �  H  H  Y  R  D  2    	  �  �  �  �  �  Y  /    �  �  �  �  �  �  e  ^  S  M  A  ,    �  �  �  k  -  �  �  I  �  �  
  _  y  e  {  �  �  �  �  �  }  g  H    �  �  �  F  �  �  -  L  W  ?  N  V  ^  ]  S  >  #    �  �  �  l  =  �  �  a  �  -  A  2  V  �  �  �  �  u  b  <  �  �  +  �  g  \  �  �    �  F  �  �  �  �  �  �  �  �  l  M  ,  
  �  �  �  �  d  ?    �  	o  	�  	�  	�  	�  	k  	E  	  �  �    8  �  G  �  )  �  R  n  ~  	s  	�  	�  	�  	x  	T  	  �  �  0  �  �    �  *  �  ;  �  Q  �  �  .  -  %      $  3  �      �  �  �  �  �  f  G    �  	�  	�  	�  	�  	�  	  	d  	E  	  �  �  C  �  [  �    q  �  j  �  $      �  �  �  �  �  y  Y  9    �  �  �  �  �  �  u  j  �  �  �  x  `  H  1        �  �  �  q  D    �  �  �  w  	X  	u  	�  	�  	y  	l  	\  	@  	  �    (  �  i    �  
    �  n  �  �    #  9  B  9    �  �  '  �    1    �  �  �  �  �  '  �  �  �  �  ~  b  H  .    �  �  �  r  E      �  �  �  �  �  m  Z  H  3      �  �  �  �  z  }  �  �  �  �  �  �  
�  
�  
�  
�          
�  
�  
h  
	  	�  	  e  �  �    <  y  �  �  �  �  x  U  0  �  �  �  b  $  �  �  �  H    �  �  f