CDF       
      obs    6   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?��+J      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M��   max       P��0      �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       �@�   max       =�F      �  \   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?s33333   max       @E�\(�     p   4   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�\(��    max       @vxz�G�     p  (�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @0�        max       @R@           l  1   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�        max       @�:`          �  1�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ��P   max       >]/      �  2X   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A���   max       B-)u      �  30   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A��   max       B-?�      �  4   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       ?I��   max       C�u�      �  4�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       ?O�   max       C�g�      �  5�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          �      �  6�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          =      �  7h   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          ;      �  8@   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M��   max       P��0      �  9   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�b��}W   max       ?۲��m\�      �  9�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       �@�   max       =�F      �  :�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?s33333   max       @E�\(�     p  ;�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�\(��    max       @vxz�G�     p  D   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @)         max       @R@           l  L�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�        max       @�:`          �  L�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         D�   max         D�      �  M�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�L�_��   max       ?۱[W>�7     �  N�   
      =   	         o      +   $            =          	               �         	            ;      �      
      ;         #            1   %               G               6   ANt��N�RPh�N�4N��iM��P6�N`��PY�P��O��N���O��P��0NaQO���N�L2N��N;N��gN�ZP�/�O�=O��NT�rOKZ�Nw�aN;�P@�O��0Pd~gN�M�O&p�O�2�P��O~/Os�_O}�=NA;�N���N&|�O��O��4N��O10|N�AOKn8O���NךkO#��N�ŖO#L�O��qO&���@��o�#�
�#�
��`B;o;��
<o<t�<#�
<49X<T��<u<u<u<u<�o<�o<�C�<�C�<��
<�1<�1<�j<�j<ě�<�/<�h<�h<��=o=o=o=+=#�
=0 �=@�=P�`=Y�=Y�=Y�=ix�=u=u=}�=�o=�o=�\)=���=���=��T=��T=��=�Fqqt|�������tqqqqqqqqot��������vtoooooooo#/<Hanrqu~mU</#,/<CHMUWUSVUPH@<66/,������������������������������������������������	�������������������������������������)BNWY_hke[N5QOOSanz���������zaUQ��������������������IKJNTamz��������zaTI��������)2:6)�����NNP[agkg`[YNNNNNNNNN
#/<HJ^aUH?</#QSTUXahnqzqnka[UQQQQ}�������������}}}}}},/<?HLLH<84/,,,,,,,,,'-08<IOPOKIB<50,,,,�����������������6BOcwrab[OB@���MNXht���������uh[VOM���
#/4<CELOD/#
����������������������_\[\`bjns��������ne_��������������������

 �����������+*/<Ganz�����zaHA<5+�������
���������� 	)BR_fmkkg[B( ��������������������������������������������)5BEW[^TB)��������)5HKF9)���
 #0003/#
�����������nlnw��������������tn��������������������913<>HNTSIH<99999999�����������������������������	�����������4=BGGB5)��[[\behptu|{xvtphc_[[	�
#0<@<;0)
	��������������������������"#"���fgo��������������znf^[\acmtz��zvmda^^^^�������	�����77BN[]bc[NGB77777777ollt������������|vto������
��������������	 ��������zÇÌÓ×ÓÎÇ�z�s�p�v�z�z�z�z�z�z�z�z�t�t�z�t�m�g�e�g�o�t�t�t�t�t�t�t�t����0�:�<�F�Q�?�/�"������������������������������������~�z�m�j�m�p�m�i�z������Ź������������������ŹűŭŤŭŲŹŹŹŹ�m�y�����y�m�`�V�`�k�m�m�m�m�m�m�m�m�m�m���������� �����������������������ù��������������üùìììôùùùùùù���������	��"�2�(�	���������i�^�g�r�����nŇŠŹ����������ŹŭŔ�{�n�b�Y�V�T�U�nE�E�E�E�F	FF!FFE�E�E�E�E�E�E�E�E�E�E�������������ŹŴűŶŹ����������������������������������������������������������Z�����������������j�Z�A�(������'�Z���������������������������������������������������������������������������������������s�f�e�f�l�s�{�����s�t�u�s�q�s�s�s�f�a�^�c�f�n�s�s�s�s�s�sÓØÔÓÎÇ�z�z�w�zÇÌÓÓÓÓÓÓÓÓ���������������������{�r�m�r�z����ƧƨƳƽƿƳƧƚƒƎƌƎƚƦƧƧƧƧƧƧ���ּۼӼ̼������r�M�'������E�Y�r�����(�4�Z�h�j�f�Z�M�E�4�(������
���(�y�����������������y�m�`�G�;�1�4�G�\�m�y���������������������������������Ľнݽ�����������ݽǽ������������ĺ����� ������������������H�U�a�i�j�a�U�M�H�D�H�H�H�H�H�H�H�H�H�H���(�B�N�\�^�X�A�5�(�������ݿҿ�����������žǾǾ���������Z�O�W�f�s�������6�O�h�}ą��t�h�[�B�)���������������T�`�a�m�p�v�m�`�T�L�G�B�@�@�G�N�T�T�T�T��"�.�7�4�4�*�"���	�������������	���f�s�u�t�w�r�k�Z�M�4�*�(���!�.�7�A�M�f�T�y�����������������m�`�T�G�A�@�B�I�Q�T�����!�.�7�.�!� �������ټ׼���	��"�.�;�G�U�]�W�J�G�;�.�"��	�������	��(�5�N�Y�Z�g�e�Z�N�A�5�(���������ǔǡǨǭǴǭǡǔǈǆǈǓǔǔǔǔǔǔǔǔD�D�EEEEEED�D�D�D�D�D�D�D�D�D�D�D�ÇÓàäìîìàÓËÇÄÇÇÇÇÇÇÇÇ�H�U�a�h�p�u�}ÌÚàÓ�z�n�U�'�� �0�<�H�=�V�b�n�a�V�A�$������������������$�=�-�5�:�F�S�]�_�`�_�S�F�:�-�"��
��!�*�-�Z�f�s�����������������u�s�l�f�Z�Y�Y�Z����������������������ĿĻĻĿ����������������
�������
�����������������񻪻��û޻�� ����
����ܻ̻�����������ŭŹ��������������ŹŭţŠśŠŤŭŭŭŭ¿��������������� ������������¿¼µ·¿��"�+�-�&�"��	���	������������������������������������������������׻��!�-�3�4�#�����ֺ��������˺׺���D{D�D�D�D�D�D�D�D�D�D�D�D{DxDpDoDnDoDsD{ / i 0 ? H e 9 R o 8 1 Y ' 6 9 B [ � b , G N % @ " [ F F C E & + R a - > : I K " L S u i 1 1 ! m % l > O X 2  �  [  �  �  �    P  �  @  �  �  �  6  �  *  G  �  �  Y  �  �  �  ]  �  `  �  �  =  �  �  �  �  �  7  a  b  �  	  ^  �  `  2  �    �     �  �  �  �  �  j  �  \��P��=L�ͻo��o;�o=�<�1=T��=8Q�=�P<��
<�=���<�C�=@�<���<��
<��
<�/<���>]/=�w=L��=o=H�9<��=C�=�-=q��>$�/=�w=,1=aG�=���=L��=�\)=� �=q��=�t�=m�h=�
==ě�=�hs=���=���=�Q�>\)=�E�=\=�j=�
=>�w><jB
VlB
8�B B�B��B-)uB/`B�ZB�Br�BN�BA���B�B��Bd�B[BS�B��B&M�B��BD�BsB��B B(��B�NBr�B<vB��Bh#B  �B"D.B_:B��B$�>BU�B
�@BB1B�B!��B�+B�B�B%#Bl�B_\B��A���B�eB(`B
��B'EB�bB
A�B
}�B��B@UB��B-?�B?�B�fB��Bg�B1[B@A�P�BA;B��B�,B�vB��B��B&@0B��B> B!�B�SB %B(@B2�ByBgzB�]BB�B��B"GB�B�SB%!B>LB
�IBA�B�&B"A�B�B�.B�2B$��BL�B��B��A��B��BB
�yB=?B�A��8A���A���A��$A�*AkZ�A�zEA�QA�GA�u�C�u�A�Q�A�i�A�S�A��PAҶ�ADz�AB��Aɠ�@��B�@�F7A8�uAj&W?I��A(��?_��A��'A���AI�A�GoAg��A]�=A<yAlyEA��A`-AA�!IB�XC�O�A��nA���B	WO@{.ADH"A�5A��@���A��6A�J�A���A���@S�LC��'AȗsA�kIA��FA�^A���Aj��A�t�A͇�A��[A�]�C�g�A��wA�m�A��;A�|KA�~�AC%A@��A�t@�x�B�@�,;A8&�Al�?O�A) �?U�NA�}�A�s AK	cA؇�AhHA]d�A>��Ame�A]�A_/�A�~LB�\C�IaA�D�A�f�B��@|*vAC�A��A�-@��A��A�o�A��>A���@T%C��}         =   	         o      ,   $            =      !   
               �         
            ;      �            <         $            1   %               G               6   B         3            -      /   %            ;                        =                     )      1         %   #                     '   '               )               !            +                  '   !            ;                                             '                  !                     #   '               )                  Nt��N�RP NG�N��iM��O�N`��Oܘ�O�%hO��N���O��P��0NaQO):MNc�N��N;N��gN�ZO��O���O��-NT�rO>�TNw�aN;�O�u1OZoO��N�M�O&p�O��O�V�N�0FOs�_O}�=NA;�N���N&|�O�i�O��4N��O10|N�AOKn8O�aWNךkN��N�ŖO#L�O�j�O&��  �  �  :  }  �   �    �  1  �  n  c  �    �  3  �  �  O  �  �  �     D  �  �  �    �  �  �  �  �  �  Y  m  �  �  U  }  �  �  �  �  E  u  �    �  |  �  �  
6  �@��o;�`B�o��`B;o=8Q�<o<�o<���<49X<T��<u<u<u<ě�<�C�<�o<�C�<�C�<��
=�`B<�j<���<�j<���<�/<�h=+=��=�-=o=o=\)=L��=8Q�=@�=P�`=Y�=Y�=Y�=m�h=u=u=}�=�o=�o=�hs=���=��
=��T=��T=�/=�Fqqt|�������tqqqqqqqqot��������vtoooooooo/<HUaqpqna</#9;<<GHJRTNH<99999999�������������������������������������������������������������������������������������)BKRU]aaYNB5)QOOSanz���������zaUQ��������������������IKJNTamz��������zaTI��������)2:6)�����NNP[agkg`[YNNNNNNNNN#/<AHMUURHD</#RTUagnpxnnma\URRRRRR}�������������}}}}}},/<?HLLH<84/,,,,,,,,,'-08<IOPOKIB<50,,,,���������������
)6DJNNJB6)NOW[ht���������thZPN��
#/2<BDJMHA/#
 ���������������������_\\]bnr{���������nf_��������������������

 �����������+16=Hanz�����zaH<9/+��������
 ������)5BLV\^^YYPB5)��������������������������������������������)5@BISSNB)�������)5@B<5)���


#(,,010,#
�����������nlnw��������������tn��������������������913<>HNTSIH<99999999�����������������������������������������4=BGGB5)��[[\behptu|{xvtphc_[[	�
#0<@<;0)
	��������������������������"#"���fhp��������������znf^[\acmtz��zvmda^^^^�������� ������77BN[]bc[NGB77777777ollt������������|vto���������������������	 ��������zÇÌÓ×ÓÎÇ�z�s�p�v�z�z�z�z�z�z�z�z�t�t�z�t�m�g�e�g�o�t�t�t�t�t�t�t�t����������'�-�/�.�$���������������������z�����������������z�u�r�z�z�z�z�z�z�z�zŹ������������������ŹűŭŤŭŲŹŹŹŹ�m�y�����y�m�`�V�`�k�m�m�m�m�m�m�m�m�m�m����������
�������������������������ù��������������üùìììôùùùùùù�������������	�������������w�l�|�����nŇŠŹ��������ŹŠŇ�{�n�b�\�[�\�a�k�nE�E�E�E�F	FF!FFE�E�E�E�E�E�E�E�E�E�E�������������ŹŴűŶŹ����������������������������������������������������������Z�����������������j�Z�A�(������'�Z��������������������������������������������������
����������������������������������s�f�f�f�m�s�|�������s�t�u�s�q�s�s�s�f�a�^�c�f�n�s�s�s�s�s�sÓØÔÓÎÇ�z�z�w�zÇÌÓÓÓÓÓÓÓÓ���������������������{�r�m�r�z����ƧƨƳƽƿƳƧƚƒƎƌƎƚƦƧƧƧƧƧƧ�f�r�����������������r�f�Y�N�H�J�Q�Y�f�(�4�M�Z�d�g�c�Z�M�A�4�(��������(�y���������������y�m�`�G�A�;�3�7�G�^�m�y���������������������������������Ľнݽ����������ݽн��������������ĺ����� ������������������H�U�a�i�j�a�U�M�H�D�H�H�H�H�H�H�H�H�H�H��(�5�A�M�Z�\�V�A�5�(���
�������������������þ¾���������������s�m�q����)�6�O�h�r�s�o�h�[�O�B�6�)�������)�T�`�a�m�p�v�m�`�T�L�G�B�@�@�G�N�T�T�T�T��"�.�7�4�4�*�"���	�������������	���M�Z�f�s�s�v�p�h�Z�M�A�4�(�#��$�1�<�H�M�`�y���������������~�m�`�T�M�H�G�H�Q�T�`�����	������������ܼۼ����	��"�.�;�G�U�]�W�J�G�;�.�"��	�������	��(�5�N�Y�Z�g�e�Z�N�A�5�(���������ǔǡǨǭǴǭǡǔǈǆǈǓǔǔǔǔǔǔǔǔD�D�EEEEEED�D�D�D�D�D�D�D�D�D�D�D�ÇÓàäìîìàÓËÇÄÇÇÇÇÇÇÇÇ�<�H�U�a�g�o�u�}ÌÙßÓ�z�U�)��!�!�/�<�=�V�b�n�a�V�A�$������������������$�=�-�5�:�F�S�]�_�`�_�S�F�:�-�"��
��!�*�-�Z�f�s�����������������u�s�l�f�Z�Y�Y�Z����������������������ĿĻĻĿ����������������
�������
�����������������񻪻��û޻�� ����
����ܻ̻�����������ŭŹ��������������ŹŭţŠśŠŤŭŭŭŭ¿��������������������������¿¾·º¿¿��"�+�-�&�"��	���	������������������������������������������������׺����!�-�1�1�!������ֺɺźɺͺں��D{D�D�D�D�D�D�D�D�D�D�D�D{DxDpDoDnDoDsD{ / i * F H e  R h ; 1 Y ' 6 9 8 ` � b , G 3 ! @ " ] F F A 1 " + R _ ( ' : I K " L R u i 1 1 ! m % ^ > O G 2  �  [  |  m  �    W  �  X  �  �  �  6  �  *  }  �  �  Y  �  �  I       `  �  �  =  b  �  v  �  �  �  �  �  �  	  ^  �  `  5  �    �     �  �  �    �  j  &  \  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  �  �  �  �  �  n  Z  G  1    �  �  �    O    �  �  t  C  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  K  �  �       6  6  #    �  �  q  5  �  O  �  3  �    �    2  H  \  o  |  x  s  i  ^  J  .    �  �  �  �  t  T  4  �  �  �  �  �  �  �  {  e  O  <  *      �  �  �  �  z  e   �   �   �   �   �   �   �   �   �   �   �   �   �   �   �   �   �   �   �   �  �  �  	p  
  
}  
�           
�  
�  
T  	�  	+  g  �  p  �  A  �  �  �  t  W  <  #        �  �  �  `  %  �  �  c    �  �    (  0  ,    �  �  �  z  :  �  �  ^    �  y  �     �  L  e  }  �  �  �  �  �  d  @    �  �  U  
  �  A  �  T  �  n  E    �  �  �  v  N  #    �  �  G  �  �  �  +    �  �  c  b  a  ^  X  R  K  D  =  3  )        �  �  �  �  �  �  �  �  �  �  y  k  Z  G  1    �  �  �  f  )  �  �  }  w  r    �  �  �  �  �  U    �  �  �  w  M    �  F  �  �  �  �  �  �  �  �  
    ,  >  O  a  n  x  �  �  �  �  �  �  �  �  �  �      .  3  /  !    �  �  �  �  �  n  0  �  )  �  �  �  �  �  �  �  �  �  �  �  t  c  O  8  2  N  a  R  B  /    �  �  �  y  a  J  4      �  �  �  �  �  �  �  �  v  g  W  O  O  O  O  O  O  O  M  I  F  B  ?  ;  9  :  ;  =  >  ?  @  �  �  �  �  �  x  j  `  X  U  T  U  b  v  i  M  7  "      �  �  �  �  �  �  �  �  �  q  \  E  /      �  �  �  �  e  
�      �  �  �  =  �  �  �  �  �  U  �  G  C    T  �  �  �  �     �  �  �  �  �  �  �  �  p  Y  >     �  �  �  z  Z  @  C  A  8  &  
  �  �  �  U    �  �  o    �  f       �  �  �  �  �  �  �  �  �  �  v  [  =    �  �  �  �    �  �  �  �  �  �  �  �  �  �  �  s  D    �  �    �  E  �  Q   �  �  �  p  ]  K  9  (      �  �  �  �  �  �  �  �  �  �  �      �  �  �  �  �  �  �  �  �  �  �  w  h  X  M  C  9  /  �  �  �  �  �  x  R  '  �  �  p  "  �  f  �  m  �  	  �    �  �  �  �  �  �  �  �  �  �  �  ]  6    �  �  }  (  �  �    6  =  9  G  W  �  �  �  �  �  �  O  �  d  
~  	9  s  �   �  �  �  �  �  �  �  �  �  {  n  `  Q  <  '    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  p  W  =  !    �  �  �  �  p  �  �  �  �  �  �  �  y  x  �  �  �  �  �  y  X  $  �  '  �    E  O  W  S  D  1    �  �  �  J  �  �  "  �  �      W  k  k  j  j  k  l  g  ]  S  D  4  #    �  �  �  �  �    ]  �  �  �  �  �  �  r  X  9    �  �  �  S    �  �  ?  �  h  �  �  �  �  �  �  �  �  �  i  /  �  �  C  �  �    �    3  U  ?  )    �  �  �  �  �  d  C  "    �  �  �  x  P  )    }  Y  3  	  �  �  x  D    �  �  d  +  �  �  ~  A    l  �  �  �  z  s  l  e  _  X  O  D  9  /      �  �  �  �  �  �  �  �  �  �  �  �  \  +  �  �  u  %  �  V  �  r  �      �  �  �  d  0      �  �  g  &  �  �  L  �  ~    �  x    �  �  �  �  �  m  [  P  G  :  (    �  �  e    �  �  `  >  $  E  (    �  �  �  �  u  T  1    �  �  �  t  T  ,  �  �  !  u  k  ]  I  3      �  �  �  �  m  T  ?  1  "    �  D  �  �  �  g  ?    �  �  �  ^    �  �  S    �  `  �      �    �  �  �  �  �  �  �  Q  >  
�  
�  
  	�  �  0  H  �  h  �  �  �  �  �  e  F  )    �  �  �  �  �  �  d  B    �  .  E  [  K  ;  x  |  l  R  3    �  �  {  D    �  |  1  �  j  �  �  ]  6    �  �  �  �  w  _  E    �  O  �  �  =   �   �   h  �  �  �  �  �  ~  n  a  J  #  �  �  s  /  �  n  �  �  �  �  	�  	�  
3  
  	�  	�  	�  	[  	  �  o    �    k  �    ?    f    �  �  \    �  k    �  C  �  C  �  
�  	�  �  �  �  �  M