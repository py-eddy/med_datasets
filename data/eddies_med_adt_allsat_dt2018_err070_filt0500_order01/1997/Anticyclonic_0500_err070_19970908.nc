CDF       
      obs    B   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?�I�^5?}       �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       N�   max       Q+       �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       �]/   max       >J       �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�        max       @FS33333     
P   �   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?��Q�     max       @v���R     
P  +   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @!         max       @Q�           �  5d   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�_        max       @�3@           5�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �L��   max       >s�F       6�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A��f   max       B1%�       7�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�Go   max       B1B�       9    	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       ?X�   max       C���       :   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       ?R�   max       C���       ;   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          �       <   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          U       =    num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          =       >(   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       N�   max       P�mW       ?0   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?������   max       ?�q����       @8   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       �]/   max       >J       A@   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�        max       @FP��
=q     
P  BH   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�=p��
    max       @v�\(�     
P  L�   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @"         max       @P�           �  V�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�_        max       @���           Wl   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         D	   max         D	       Xt   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��ߤ?�   max       ?�p:�~�      �  Y|               b   �                     
                  H   �         D   ,   �   '   O   	      G   8         !            
                        
      �      
   !               6      �            �         .   N���N���N^hZN��P�T�Q+Om�iO&O�7�O�ҺN�{XO��0N���O��OR_yN��EO�:N���PWI$P� @O�N�O���P�YP�f$O£dP�hO\�Of��P�fRP�OߋNwڻP*��NA0N��/N�WEN{,N��}N"B�O�wO�jN��Ne�N�N7��O�~�OƱNm��N�MO�EXOpR�O
LO�N���O��|N6L>P(
+N��nN��N8*�P��N��N�wOp\�NJ%3�]/�+�49X�49X�#�
�#�
��o��o;ě�<t�<t�<T��<T��<e`B<u<u<�o<�C�<�C�<�C�<�C�<�t�<���<��
<�9X<�9X<ě�<ě�<ě�<�`B<�`B<�`B<��=o=o=\)=\)=�P=�P=�P=�P=�w=�w=�w=�w='�=,1=,1=,1=8Q�=<j=T��=Y�=Y�=aG�=e`B=ix�=q��=q��=}�=��=��w=�1=�j=��>J��������������������plkpt�������ytppppppvz������������vvvvvvrnpt�����trrrrrrrrrr�����5Q[kmkN�����?Ba}���46,����qJ?t�����������������|t #*0<FIUWYXUMI<0#�����������������
"/8FIIHE;/"��������������������hr��������������ysgh��������������������.1>CHUX_adeggaWUH<3.��������
"
�����}������������������������
#% 
�����)+6;86)  ))�������������������)B[�������t[5��7249<HLU_amfaUU[UH<7��������������������)6BKO\bXNB6)����)BKN=&�����zzw��������������{z���
#%&$!
�������^afrz�����������zmf^��������������������
#%)00/*&#
����
")5BRQ81<9)���FGK[h�����������h[OF��������������������HEHHIUanxunaUHHHHHHH�����)-5BRfD5�������������������������	��������������������STYamsumidaZ[TSSSSSS��������������������������������$**6CO\ghqlh\OJC:6*$�
#02<BC<60#
��DGLN[ggc[NDDDDDDDDDDNJO[fhqh[ONNNNNNNNNNwu����������wwwwwwwwXbhnw{����}{nbXXXXXX)##+/;HTYafihaTH>3/)�������

�����B:;BO[\^[TOBBBBBBBBBA<;=BOQ[b[YSOBAAAAAA*).6<L[huvvxtoh[OB6*RTVVZ_amz�����zmaVTR������������������������
#**&#
�����GGHIU`aihaUHGGGGGGGG�����5?AB@5"
�����������
,+#
�������������������������//:<=HJTH<://///////��������������������|z�������	
�����|��������������������-./:<<?A?<3/--------.+##;IJF?<50/.a^UHFCFHUYaaaaaaaaaa�����������������������������������������n�zÇÓ××ÓÇ�z�n�a�^�a�e�n�n�n�n�n�n�N�[�`�[�U�N�D�B�7�5�3�5�B�C�N�N�N�N�N�N�@�L�Y�_�`�Y�L�@�<�=�@�@�@�@�@�@�@�@�@�@�<�I�b�{ŇŔŎŀ�n�c�<�0������������<�s��������<�=�0���������s�Z�[�L�)�A�Z�s��"�'�)�'�������ܹϹù��ùϹܹ������������������������������~�x�v�}�w���������ûлػܻ���ܻۻλ�����������������"�/�H�T�\�_�T�H�;�/�"��	��������	��s�v�x�u�s�f�c�Z�M�A�5�A�K�M�Z�d�f�l�s�s��������������������s�m�f�a�_�a�f�s���F=FAFJFPFVF\FVFUFJF=F1F1F'F1F2F3F=F=F=F=�a�m�����������������z�m�a�T�H�F�D�L�T�a�������%�)�,�*�!����������������������)�$����� ����������Y�f�����������������t�f�J�@�5�/�4�M�Y�@�3�'��������
����'�-�/�5�@�@àù������������ùìÓÇ�z�C�0�H�U�aÇà��)�B�hąęęĈ�t�[�B�)������������������������������������������������������������������~���������������������ʼּ߼�����ּ����������������������������������������z�m�c�d�u�������������!�,�M�T�L�:��⺤����������������B�M�P�\�]�O�A�4�������������@�B�t�xĚĳĺľ����ĹĳĚā�h�a�[�N�J�O�[�t���������������������������������u�m�s���׾�����"�,�+�"������׾ʾ��¾ʾϾ�ƎƳ������$�C�N�2����Ƨ�u�*�����*�OƎ�f��������������������s�Z�R�P�C�;�@�K�f�����	�	������׾ʾƾ������Ǿʾ׾���zÇÓÓÓÈÉÎÇ�z�q�s�w�z�z�z�z�z�z�z�������	�&�.�.�!�������������������������/�<�@�?�<�8�/�$�#��#�-�/�/�/�/�/�/�/�/�.�;�G�T�`�m�o�m�`�^�T�G�;�.�#�)�.�.�.�.àìùþúùìèàÓÇÅÇÇÓ×àààà��������������ƶƳƧơƧƳƾ�������������-�:�F�O�S�_�l�n�x���x�l�_�S�F�:�-�,�)�-������	��������	�����������!�"�-�,�(�"���	������������	��������
����������ټҼּؼ����ݿ���������ݿտؿݿݿݿݿݿݿݿݿݿ��/�<�H�C�<�8�/�,�,�/�/�/�/�/�/�/�/�/�/�/�/�<�>�G�C�<�4�/�,�$�&�,�/�/�/�/�/�/�/�/���������������������������������#�0�<�I�L�S�U�R�I�;�0�#��������
���#D�D�D�D�D�D�D�D�D�D�D�D�D�D�DzDpDqDwD{D��ѿݿ�������ݿٿѿοпѿѿѿѿѿѿѿѺ������	����������ܺں������������'�4�7�4�'����ܻ׻ӻԻл߻�ŇőŠŭŹ����������ŹŭŠŔņ��y�{�}Ň�����������������������������������������������������������y�m�`�[�W�_�m�n�y�������%�)�3�)�(��������������������
������������¶¦¦²¿���g�o�t�}�t�g�[�N�D�N�[�]�g�g�g�g�g�g�g�gE\EiE�E�E�E�E�E�E�E�E�E�EiE\EVEWETELEJE\�y�����������������y�t�n�l�f�l�p�y�y�y�y�(�*�4�?�4�,�(�����'�(�(�(�(�(�(�(�(�
�����
������������
�
�
�
�
�
�
�
�'�4�T�a�f�_�M�4���������������'�������������������������������������������!�*�*�*���������������ĳĿ�����������
���
������������Ŀĳĳ��ŹŮŴŹ������������������������������ - G ^ 4 + L V @ + * � . : J  5 A T > D H {  C / 4 1 W T | : A | I N d Q O � x g = ? k g � >  # 6 7 : ; b '  t 9 2 d  g  ^ 8 -  �  �  �  7  �  	�    S  m  W    '  �  /  �  �  �    t  �  6  Z  �  �  �  �  �  �  �  	  �  B  �  �  H  �  �  �    x  p  v  �  J  �  �  )  �  v  �  m  �  +  b  �  7  m  �  �  X  �      O    \�L�ͼ�1��`B���
=�9X>+<���;ě�=�P<�<D��<�`B<�j=�P=+<�t�='�<�h=�E�>s�F=,1<�9X=� �=�%>;dZ=u=��=o=<j=ȴ9=�1=#�
=�P=��=��='�=u=<j=0 �=#�
=@�=aG�=0 �=,1=L��=L��=�\)>E��=H�9=aG�=���=��=��=��P=��=�;d=�%>?|�=��=�+=�t�>T��=ȴ9=Ƨ�>��>O�B�*B
#B
�B��B��Be^Bx�B&,�B"�A��fB�.BVB��B_4B>*B8�B#�B��B_B �B.�B_�B��B�B�jB#ܫB
�Bf:B]�B��B~�B ��B�B�B�.B��B"Q�A���B+NOB.;�B1%�B%.kB�#BD�BmB(b�A���ByB��B��B A��jB �BҭB�{B�MB�XBZ�B,RzB�jB~^B%�B��B��B߈B�BH�B
�B
[@B�nB�rB��B�VB&`B#1�A�GoBJ3BJ	B�<B?�B=6BA^B#��B�B�KB?�B?�B5�B�sB,�B�<B$ #B74B�5BE$B��BB�B �rB�B�.B�B>�B"@A���B+S�B.��B1B�B%A�Bf�BlBe�B(�)A��5B?}B�BO1BFUA�J�B4�B<nB��B��B�PB�4B,E�BƂB�pB��B�NB�B�5B�2A�g�Aȧ�A��.?ʯA��A�*$?X�@���@��A�!nA>��AE�gC���A�\�A��aAԚ�@���?�ZIA�w�A�:QAҔfA�4)@�`�A�CI@C�dA6��Aޠ�A��[AXq�B�ZAD
vAT��A�_�A�u�A��JAeOVA˔#B�.@���A[U�A\?�AT�A~�zA�&|A²�@[��A�_�C���A}��@O��@�ZuA��NAs%hAl�<A�I�A�-�A���C�zAM~A6� A��@�4A���A���A�U�A�N�A�|�A�|�A���?��A�m�A��j?R�@��\@��:A���A@##AE��C���A���A��UA�Ø@�V?��A�o!A�u�A��A���@�u+A���@D��A5�Aި�A���AX� B��AD�MAS6�A�a�A��A�}5Ac&XAˁAB�,@�NsAZ�A\�AҿA~��AÆ�A4@\YA�~eC��}A}�@P{j@�wA��Ar��Ak�9A�s�A���A��RC���A�qA6��A��S@��:A�c}A��(A�LA�o�               c   �                                        I   �         D   ,   �   '   P   	      G   9         "            
                        
      �         "               6      �            �         /                  7   U                                 #      -   9            '   9   #   '         K   '         +                                                                  #      )            -                              7                                       #   #               !               =            +                                                                        !            +            N���N���N^hZN��O��XP���N�1O&O9]�O�ҺN�{XO��0N�d�Ob�^Ok�N��EO6�wN��O��KO���N[q�N�OO�6O��,O���O�>�O�I�O\�O%��P�mWO�4�OߋNwڻP�NA0N��/N�F
N{,N��}N"B�N��O�jN��Ne�N?c�N�MO%��O/��Nm��N�MO�EXOpR�O
LN�M�N���O�d�N6L>O���N7��N��N8*�Pc�N��N�wO,��NJ%3  V  D  9  �    �  =  ^  &     �    �    �  �  �  �  S  c  �  q  	�  �  �  $  
  �  �  ]  �  C  �  v      �  b  �  �  �  �    )  �  �  �    �  }  �  �  �       Q  �  �  C  �  �  �  ;  9  �  ��]/�+�49X�49X=8Q�=0 �;�o��o<�C�<t�<t�<T��<e`B<�o<��
<u<���<���=#�
=��<���<�t�=0 �<��=Ƨ�<���=<j<ě�<�`B=�w=<j<�`B<��=��=o=\)=�P=�P=�P=�P=��=�w=�w=�w='�=,1=L��=�
==,1=8Q�=<j=T��=Y�=aG�=aG�=�%=ix�=���=�o=}�=��=�Q�=�1=�j=�>J��������������������plkpt�������ytppppppvz������������vvvvvvrnpt�����trrrrrrrrrr����).7<=4����`y���������������nc`~|��������������~~~~ #*0<FIUWYXUMI<0#������������������
"/8FIIHE;/"��������������������hr��������������ysgh��������������������5/2<?EHUU]acdff_UH<5������

�������}��������������������������
����� )*36865)%��������������������%5BN[gr||ug[5)68<HPU[USH=<66666666��������������������6BHNPOLBA6)����)/5BD;/)����������������������������
!#%" 
������xptx��������������zx��������������������
#'*..,&#
����)5HB4.-4)���XPPRX[ht��������th[X��������������������HEHHIUanxunaUHHHHHHH���)5HPKB5���������������������������	��������������������STYamsumidaZ[TSSSSSS��������������������������������,-6COZ\ehojh\OKC<6,,�
#02<BC<60#
��DGLN[ggc[NDDDDDDDDDDNJO[fhqh[ONNNNNNNNNN����������^binx{����|{nb^^^^^^8205;HT_adca`WTOHC;8��������	

����B:;BO[\^[TOBBBBBBBBBA<;=BOQ[b[YSOBAAAAAA*).6<L[huvvxtoh[OB6*RTVVZ_amz�����zmaVTR������������������������
#()%#
�����GGHIU`aihaUHGGGGGGGG����)4;=>;5)��������
"$"
��������������������������//:<=HJTH<://///////���������������������~������������������������������������-./:<<?A?<3/--------#/6<>B?<:2/#"a^UHFCFHUYaaaaaaaaaa�����������������������������������������n�zÇÓ××ÓÇ�z�n�a�^�a�e�n�n�n�n�n�n�N�[�`�[�U�N�D�B�7�5�3�5�B�C�N�N�N�N�N�N�@�L�Y�_�`�Y�L�@�<�=�@�@�@�@�@�@�@�@�@�@�0�<�I�U�]�b�`�[�U�I�<�0�#������#�0�����������������������z�j�f�Y�[�i�s�������� ���
������������������������������������������~�x�v�}�w�����������ûлӻջлû�����������������������"�/�H�T�\�_�T�H�;�/�"��	��������	��s�v�x�u�s�f�c�Z�M�A�5�A�K�M�Z�d�f�l�s�s��������������������s�m�f�a�_�a�f�s���F=F?FJFLFVF[FVFRFJF=F4F1F.F1F5F7F=F=F=F=�T�a�m�������������������z�m�a�T�H�E�M�T������$�!��������������������������)�$����� ����������Y�f�r�����������������r�f�M�D�D�M�U�Y��%�'�-�3�3�:�3�'�!�����������n�zÇàìù��������ùìÓ�z�]�R�P�Y�a�n�)�6�O�h�zĂĄă�}�t�h�[�B�6�0�.����)������� ���������������������������������������������������~���������������������ʼҼڼټּʼ������������������������������������������������z�v�s�s�z�������ɺֺ���������ֺɺ��������������ɾ4�A�H�N�Y�Z�M�G�4�������������4�hāĚĦĳĹļĹĳĦĚčā�s�l�i�g�d�f�h���������������������������������u�m�s���������"�$�"� ��	�����׾˾ʾ׾޾�ƎƳ�������������ƧƎ�u�*����*�OƎ�Z�s�������������������s�f�a�R�M�L�U�Z�����	�	������׾ʾƾ������Ǿʾ׾���zÇÓÓÓÈÉÎÇ�z�q�s�w�z�z�z�z�z�z�z���	��(�(������������������������������/�<�@�?�<�8�/�$�#��#�-�/�/�/�/�/�/�/�/�.�;�G�T�`�m�o�m�`�^�T�G�;�.�#�)�.�.�.�.àìùüùùìæàÓÇÇÇÈÓÙàààà��������������ƶƳƧơƧƳƾ�������������-�:�F�O�S�_�l�n�x���x�l�_�S�F�:�-�,�)�-������	��������	�����������"�*�+�'�"� ���	������������	��������
����������ټҼּؼ����ݿ���������ݿտؿݿݿݿݿݿݿݿݿݿ��/�<�H�C�<�8�/�,�,�/�/�/�/�/�/�/�/�/�/�/�/�<�<�E�B�<�1�/�.�&�(�/�/�/�/�/�/�/�/�/����������������������������������#�0�<�C�G�@�0�-�#���
��������
��D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D��ѿݿ�������ݿٿѿοпѿѿѿѿѿѿѿѺ������	����������ܺں������������'�4�7�4�'����ܻ׻ӻԻл߻�ŇőŠŭŹ����������ŹŭŠŔņ��y�{�}Ň����������������������������������������������������������y�m�`�]�Y�`�m�p�y�������%�)�3�)�(������������������
��������������¿²©ª¶¿�����g�o�t�}�t�g�[�N�D�N�[�]�g�g�g�g�g�g�g�gE�E�E�E�E�E�E�E�E�E�EuEiEcEaEcEiEnEuE�E��y���������������y�x�p�w�y�y�y�y�y�y�y�y�(�*�4�?�4�,�(�����'�(�(�(�(�(�(�(�(�
�����
������������
�
�
�
�
�
�
�
�'�4�L�Z�a�[�M�4�'����������������'�������������������������������������������!�*�*�*����������������������
����
� ������������������������ŹŮŴŹ������������������������������ - G ^ 4   @  @  * � . N I " 5 = [ 1 9 3 {  (   ) ' W K z , A | L N d M O � x f = ? k Z � 5  # 6 7 : ; \ '  t + L d  c  ^ ) -  �  �  �  7  L  ;  �  S  �  W    '  �  �  ,  �  �  �    I  m  Z  �    �  u  �  �  �  �  x  B  �  �  H  �  �  �    x  6  v  �  J  �  �  j  o  v  �  m  �  +  ?  �  �  m  �  b  X  �  �    O  p  \  D	  D	  D	  D	  D	  D	  D	  D	  D	  D	  D	  D	  D	  D	  D	  D	  D	  D	  D	  D	  D	  D	  D	  D	  D	  D	  D	  D	  D	  D	  D	  D	  D	  D	  D	  D	  D	  D	  D	  D	  D	  D	  D	  D	  D	  D	  D	  D	  D	  D	  D	  D	  D	  D	  D	  D	  D	  D	  D	  D	  D	  D	  D	  D	  D	  D	  V  X  Z  \  ]  a  k  u    �  �  �  �  �  �  �  }  w  r  m  D  =  6  /  (            �  �  �  �  �  �  �  �  �  �    9  5  0  ,  (  "        �  �  �  �  �  �  �  �  p  U  :  �  �  �                        +  J  e  q  ~  �    �  �  )  }  �  �  �  �        
  �  �  �    m  L  �  �    `  �  �  �  �  �  �  �  �  �  x  #  �  !  �    �  �  �  �  �  �    1  ;  =  :  3  %      �  �  �  �  b  .  !  ^  Q  C  6  *        �  �  �  �  �  �  �  �  �  �  �  �  �  �  �        !  %  #    �  �  n  )  �  �  k  	  |  �     �  �  �  �  �  �  �  �  �  �  ~  g  J  &  �  �  �  Y  �  �  �  w  m  c  X  N  E  >  6  /  '       �  �  �  v  O  )    �  �  �  �  �  �  �  �  �  }  n  [  E  *    �  �  :   �  �  �  �  �  �  �  �  �  �  �  z  g  M  0    �  �  a  #   �            �  �  �  �  s  H    �  �  g    �  F  �  h  <  V  h  t  ~  �  �    o  X  :    �  �  �  R    �  ~   �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  }  s  i  _  U  K  5  Y  r    �  �  �  �  �  �  w  _  <    >    �  �  y  �  �  �  �  �  �  �  �  �  }  ]  9    �  �    �  {  -   �   �  �  �    4  E  N  S  L  6    �  �  6  �  o  �  R  e  3  [    |  e  (  �  }     Q  _  8  �  �    J  +  �  J  B    �  �  �  *  k  �  �  �  �  �  �  h  7  �  �  }  /  �  '  �  �  q  d  W  I  <  /  !      �  �  �  �  �  �  i  N  2     �  �  	)  	J  	j  	�  	�  	�  	�  	�  	n  	   �  c  �  |  �  �  �  x  ;  #  ?  \  �  �  �  �  �  l  D        �  �  �  O  �  �    Z  O  �  	�  
�  N  �  D  �  �  �  �  �  @  �  S  
h  	_    �  m  �  !  !       �  �  �  p  D    �  �  |  8  �  x  �  v  �  	  	z  	�  	�  
  
  
  
  	�  	�  	�  	[  	  �  �  &  x  k    >  �  �  �  �  �    f  L  0    �  �  �  �  n  _  U  @     �  �  �  �  �  �  �  �  �  �  �  j  J  %     �  �  g     �  �    (    ]  I  '  �  �  �  {  �  g  U    �  :  �    �  �    E  s  �  �  �  �  �  �  �  �  �  d  *  �  �  !    �  G  C  A  :  +      �  �  �  �  �  h  E  &    �  �  �  l  L  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  V  d  m  v  q  ^  ?    �  �  �  |  `  ?     �  Q  �  L  N        �  �  �  �  �  �  �  �  o  U  9    �  �  �  �  o       �  �  �  �  �  �  �  �  �  �  q  \  D  +     �   �   �  �  �  �  �  �  y  X  ,  �  �  �  Y    �  H  �  v  �  (  t  b  `  ^  j  x  �  �  �  �  x  l  ^  N  3    �  �  }  O  "  �  �  �  �  �  �  �  z  n  b  T  F  7  (      �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  p  P  0    �  �  �  �  �  �  �  �  �  �  k  Y  K  =  ,       �  �  �  E  �  �  =            �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  )        �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  p  Z  D  /      �  �  �  �  �  P  +  
  |  �  �  �  �  �  �  �  g  K  0    �  �  ?  �  �  ;  �  �  �  �  �  �  �  �  �  �  �  �  �  b  0  �  �  7  �    �   �  t  �    d  �  �  �        �  �  �  T  �  o  �    
  {  �  �  �  �  �  �  �  �  �  �  p  Z  @  %  	  �  �  �  �  p  }  u  l  a  U  E  2      �  �  �  �  f  B    �  �  -  �  �  �  �  �    Q  !  �  �  �  P    �  ~  "  �  b    �    �  `  V  F  6  "       �  �  �  �  T    �  s  -  �  �  �  �  �  �  �  �  �  �  �  x  ^  >    �  �  �  }  V  ,  &  :  �      	  �  �  �  �  P    �  �  Q  '    �  :  �  `  o     
  �  �  �  �  b  :    �  �  u  ?  	  �  �  S  	  k  �    B  O  P  K  A  1      �  �  �  V    �  b  �  &  ^  �  �  �  �  s  d  U  F  7  (      �  �  �  �  �  u  R  /    .  �  A  �  B  v  �  {  T    �  V  �  !  N  N  	�  E  y      !  )  0  6  <  @  B  A  9  .       �  �  �  �  K    �  �  �  �  �  �  �  �  �  �  �  �  o  X  A  *    �  �  �  �  �  �  �  �  |  h  P  7      �  �  �  �  c  0  �  �  �  G  �  �  �  �  �  �  �  J  6    �  8  �  �  ,  ^  
Y  �  �  �  ;  1  &        �  �  �  �  �  �  z  Q  .    �  �  �  6  9  -  !      �  �  �  �  �  �  �  �  �  o  ^  )  �  �  n  w  q  �  �  �  �  �  k  @    �  �  8  �  t    �  �  O  �  �  �  �  {  a  E  '  	  �  �  �  u  I    �  �  y  ;  �  �