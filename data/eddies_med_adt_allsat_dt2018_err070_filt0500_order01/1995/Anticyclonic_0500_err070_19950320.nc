CDF       
      obs    ;   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?�
=p��
      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M�cr   max       P�V      �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       �ě�   max       >+      �  �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?+��Q�   max       @E��Q�     	8   p   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?��\(    max       @vp(�\     	8  )�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @/         max       @O@           x  2�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @ˣ        max       @�ڀ          �  3X   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ��t�   max       >["�      �  4D   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�bX   max       B,�h      �  50   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�~{   max       B,�@      �  6   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       ?(W�   max       C�`)      �  7   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       ?-��   max       C�g/      �  7�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          �      �  8�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          G      �  9�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          9      �  :�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M�cr   max       P��E      �  ;�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�o���   max       ?�o hۋ�      �  <�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ��j   max       >+      �  =|   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?+��Q�   max       @E��Q�     	8  >h   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?��\(    max       @vp(�\     	8  G�   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @'         max       @O@           x  P�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @ˣ        max       @��           �  QP   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         @�   max         @�      �  R<   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�`�d��8   max       ?�o hۋ�     p  S(               S   	   Y      R               
   
   
         E   1      :   7            	                                     �                  +         	         .   9   O                  N�\�NrN�DN`ՠP�VN�1P3_O��P]L'N��7N>Oq�zO�eN�)�No�,N7
�N�uO�oP��KP�Pvr�P�;P�O���N�S]Nb#�N�7�O'�tO* �N��iN�<�N���N!�N�P�O1�N�|rO~�O�j�Pn=N�EN�L�O3�M�crN7�O�K�NzDN���NZ3�N�� N�/�O��O���PB�pM�U4N^$O;OηO$��NQp��ě���j�e`B�T���#�
�#�
�t���o�o�o��o$�  ;D��;�o;ě�<o<49X<49X<49X<D��<e`B<u<u<�C�<�C�<�t�<���<�1<�9X<�j<ě�<���<�<��=C�=C�=C�=\)=\)=\)=�P=��=#�
='�=,1=,1=0 �=49X=@�=D��=D��=T��=Y�=]/=e`B=ix�=m�h=��>+��������������������pqotw����tppppppppppffgotx|~tgffffffffff����


������������)Bt���z_N5������������������������%$;N[gpvy���t[NLB9*%�������������������,/$��������� 

�������������������������������
#/49;8/#
��}}�����������������}������mjst������|tmmmmmmmm@?@BMOP[[[YOGB@@@@@@
)25>65*)mr{��������������vqm�����)BMNF6
��������WZc��������������bWWUt����������tdjinkWnou���������������zn�������).)�������������
! �����fefht������tttkhffff/-../03<GHKHG<4/////#/4663/'# ��������������������XXZ[^gt�������ytg_[X���������������������������������������

 ������/-/68;;=GB;/////////JHKO[dhptzyvth[OJJJJ��������������������[\[ZVU[^htz}|zvtlh[[�����������������������)6AJOB6 ��������)6JQ[t~t[O6)��

#/-#










)5<540*)���������������������������������������������������������"$/<HQYZWZYLOH<551%"^^Xantz~�zna^^^^^^^^���
#),)#
���aanz����znmaaaaaaaaa���

		�������������������������	)5BPRRMB5)������'.-)����myz~��������������pm)+686)))))))))))������������������������������������VUZaclmz������zrmaVV�{|��������������������������������������6�B�O�Y�Z�O�B�>�6�0�.�2�6�6�6�6�6�6�6�6�U�a�n�r�n�d�a�U�L�I�U�U�U�U�U�U�U�U�U�U���!�*�-�*����������������������	��������������������������<�I�bœŠŞŊ�y�n�b�I�0��������������<�/�<�H�Q�J�H�<�:�/�+�$�$�/�/�/�/�/�/�/�/��6�O�\�_�_�O�)���������úóù��������(�A�T�Z�_�`�_�W�N�A�5�(������������6�B�,�'���ùìÓÊÃÄËÕà������������� ����������������������'�3�7�4�3�'��������������`�m�y������������y�m�`�T�G�>�7�<�O�]�`���4�A�B�F�?�<�4�(��ԽĽ��Ľн���ùú��������������ùîìëàØàãìøù�L�Y�e�k�o�q�e�Y�X�M�L�K�L�L�L�L�L�L�L�L�l�y�����������������y�o�l�f�l�l�l�l�l�l�Z�f�j�j�l�f�c�Z�M�I�M�M�P�Y�Z�Z�Z�Z�Z�Z�(�A�Z�g�y�����s�g�Z�N�A�5���������(�"�;�Q�]�b�S�A�"�	���������m�s���������"��Z�����������f�d�A�������	���!�������þ�����M�4�(������4�A�Z�f�s���hāĦĺ��ĿķĳĦĚčā�q�n�h�g�k�i�d�h�������Ŀѿڿؿ̿Ŀ��������y�m�h�j�p�y��D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D{DxDD�D�����������������������s�s�s������������N�Z�g�s�v�s�g�c�Z�W�N�C�A�:�A�M�N�N�N�N���������������������������}���������������������������������������y�p�m�r�y�}���m���������������������~�y�n�`�Z�X�`�f�m�"�.�;�G�T�`�a�`�W�T�G�;�.�"���"�"�"�"�����!�"�-�:�-�!�������������'�.�4�@�J�M�U�M�@�4�'�%��������/�;�?�H�T�U�T�H�;�/�%�/�/�/�/�/�/�/�/�/�ܻ��������������ܻۻһԻٻܻܻܻ������	��"�/�2�1�/�&�"��	�����������������!�-�:�C�F�O�N�F�:�-�!������ù����������ùìàÓÑÇÆÇÍÓàìñù�����ʾ����������־Ͼž��������������f�r�����ռ��ּ�������i�p�k�K�?�<�M�f�����������������������������������������B�N�[�e�g�p�k�g�_�[�N�B�@�;�B�B�B�B�B�B�y�����������������������������{�y�w�s�y�ѿҿ׿ҿѿĿ��ÿĿпѿѿѿѿѿѿѿѿѿѹ���������������ܹ׹ܹ�����������������������ùìÓÇ�~ÅÓàì�����a�n�z�Á�z�s�n�a�]�[�]�a�a�a�a�a�a�a�a��������������������������w�w�w��������<�<�E�B�>�<�/�%�%�/�/�;�<�<�<�<�<�<�<�<���������������ּӼּ�������6�C�O�W�\�h�j�t�u�x�u�h�\�W�O�<�6�1�2�6ƁƎƳ��������������ƳƚƎƁ�y�t�t�v�zƁ�����������������������������}������������!�:�F�O�O�Y�W�L�:������غֺں����@�3�'�'�"� �#�'�1�3�@�A�@�@�@�@�@�@�@�@E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�������������
��
���������������������ŔŠŭŶŹ����������ŹŭŠŝŗŔŒőŔŔ�������������������r�k�f�]�f�j�r�������E7ECEPE\E]E\EXEPECE>E7E3E7E7E7E7E7E7E7E7 O h 9 R # ? < ( \ k B B 6 U V @ c L 5 b R 9 - ' ? o * ( , } z S c 7 I ? 9 q F l N 7 | n c : ' o B Y  4 ( h S M E l 2    �    8  u  �  �  F  F  g  �  #  �  -  �  �  e  �  �  �  
  �  d  Z  �  �  �  �  b  }  �  �  �  i     �     j  1  �  g  �  t  Z  �  �  �    �  �  C    ,  7  A  {  l  W  �  n��C���t��o�#�
=����o=��
<�j=���;�o;��
<�9X=+<T��<u<�t�<�o<�h=���=y�#=C�=�t�=�O�>t�<�j<�j<�`B=�w=��<�/<�<�=C�=<j=L��=]/=�+=q��>["�=�P=0 �=L��=,1=<j=�1=D��=aG�=Y�=aG�=}�=��=�;d>$�=q��=�C�=�\)=�\)=��>�+B��B	��B	��B$#eB�B�hB~�B�?BkBM-B!v�BT	B �BB�B�hB��Br�BO_BFBrB!jB�?B�gB
(BW�B�)BҀBŌB	�Bs�B �B#ȅA�bXBy1B��B�}B!ԻB��B�2B$�B�bB,�hB]DB��B�NB��B$��BpB$A�B�B��B.{B��B/�Be�BSuA��%Bx�B@�B��B	ķB	��B$(B��B��BAB�B}�B?(B!D�BCB ��B..B�EB�9B�kB�Bj�B
̛BBzB�B��B6�BD<B�B�#BښB	��BAB ��B#̚A�~{B�@B��B�JB!�BŌB�UB%<BS�B,�@B�>B��B�2B��B$M�BB@B$@�BϗB�SB�B\zBD}B?`BB�A��CB�eBA�A�g�A�CA�$�A1�lA�1>A��A��A�k�A��A�G�?��;Aij�A28�A�u�?�?�A��A?�\A�geA��A�!�AAd�A��As�C��AH*�A�w�A���Ap�2Am�hAcX�@g�$@��RA��u@�<A�S�@w!`Ă�ANT�@��AIy�A��A WjAz�e?(W�A�t�A�p�@�o/A��A)�BmKB�A�5_@b<�?��}C�`)A��A�VP@�F�C���A��AŅA�~?A1�wA�w`A�\�AӅA�XzA��8A���?��WAi$A0�A��?�A�A>ۡA���A���A�oAC��A�w�Ar�CC��%AGH�A��*A�8Aq&>An�dAb�@j \@��A�]�@��A�|�@{k�A�y�AR��@���AI~�A��)A70Az�?-��Aч�AƝ�@���A�A�B;'B��A���@[�?��LC�g/A��#A��@��-C��Y               T   
   Z      R               
   
            E   2      :   7   �         
                              !      �                  +         
   	      /   :   O                                 A      -      9            '               '   G   G   7   #   %                                             %   7                  !                  #   #   -                                 9                                          7   -   3                                                   %   #                                          !                  N=3�NrN�DN`ՠP��EN�1OU�9O9��O��kNUDN>O*.O�9N�)�NE0N7
�NU��O���Pss+P9�PbNO��O�kO�N�S]Nb#�N��O	4ANؙ�N��iN�<�N���N!�N�P�O1�N�P�N�iO�j�O��JN�EN�L�O3�M�crN7�O)J$NzDN���NZ3�N�� N�/�O�NJO���O�M�U4N^$O;OηO$��NQp�  �  *  i  �  N    	*  �  	_  &    |  �  r  �  �  0  {  	  �  I  �  o  �  �  �  �  �  �  �  "  [  O  P    �  X  V  ,  3  M  �  )  �  >  �  �  �  2  4  `  �    �  �    �  `  <��9X��j�e`B�T��<o�#�
=�P;�o=+��o��o;��
<D��;�o;�`B<o<D��<�o=+<�j<u=+=\)=��-<�C�<�t�<��
<�j<���<�j<ě�<���<�<��=C�=#�
=#�
=\)=�-=\)=�P=��=#�
='�=Y�=,1=49X=49X=@�=D��=Y�=y�#=��-=]/=e`B=ix�=m�h=��>+��������������������pqotw����tppppppppppffgotx|~tgffffffffff����


�������������)B]lmYN>5������������������������@;98<BN[\gkpqng[NGB@�����������������������������

���������������������������������
#%/562/#
 ����������������������������qktt������xtqqqqqqqq@?@BMOP[[[YOGB@@@@@@)/5<5)ttv}��������������zt������6CC6�������dbct�������������tgdYW[t���������xglkpmY������������������������������������

	��������fefht������tttkhffff/-../03<GHKHG<4/////#/2541/#��������������������][[`gt{����{tg]]]]]]���������������������������������������

 ������/-/68;;=GB;/////////JHKO[dhptzyvth[OJJJJ��������������������ZY[dhtuxxutha[ZZZZZZ�����������������������)6AJOB6 ���������)6?DLG6)��

#/-#










)5<540*)���������������������������������������������������������+)'(,/3<HJSSPMKH<4/+^^Xantz~�zna^^^^^^^^���
#(+(#
�����aanz����znmaaaaaaaaa���

		������������������������)5BMPPNHB5)�����#(*+)������������������������)+686)))))))))))������������������������������������VUZaclmz������zrmaVV�{|��������������������������������������6�B�O�R�R�O�B�9�6�3�0�5�6�6�6�6�6�6�6�6�U�a�n�r�n�d�a�U�L�I�U�U�U�U�U�U�U�U�U�U���!�*�-�*����������������������	��������������������������#�<�I�bŇŋŋ�|�n�b�U�
�������������
�#�/�<�H�Q�J�H�<�:�/�+�$�$�/�/�/�/�/�/�/�/������)�1�6�,�)���������������������(�5�7�A�N�R�V�U�N�K�A�5�(�$�����������������������������ùìàÛÚÞì��������������������������������������'�3�7�4�3�'��������������m�y�~�������{�y�p�`�T�G�D�<�B�G�V�`�j�m��(�4�7�8�4�(������׽ֽݽ�������ùú��������������ùîìëàØàãìøù�L�Y�e�f�m�n�e�Y�Y�N�L�L�L�L�L�L�L�L�L�L�l�y�����������������y�o�l�f�l�l�l�l�l�l�f�i�h�j�f�a�Z�P�O�R�Z�^�f�f�f�f�f�f�f�f��(�5�N�Z�d�g�k�g�e�Z�N�5����������"�/�;�L�Q�U�N�0��	�����������������
�"�5�N�g�����������e�W�U�A�(����"�)�/�5��������������M�4�(�����4�A�Z�f�s��āčĚĦĪĳĺĸĶĲħĚčā�y�p�q�s�~ā�������ĿƿĿ��������������y�v�v�|������D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�����������������������s�s�s������������N�Z�g�s�v�s�g�c�Z�W�N�C�A�:�A�M�N�N�N�N�����������������������������������������������������������������y�s�p�w�y�������m�y���������������y�u�m�`�a�m�m�m�m�m�m�"�.�;�G�T�`�a�`�W�T�G�;�.�"���"�"�"�"�����!�"�-�:�-�!�������������'�.�4�@�J�M�U�M�@�4�'�%��������/�;�?�H�T�U�T�H�;�/�%�/�/�/�/�/�/�/�/�/�ܻ��������������ܻۻһԻٻܻܻܻ������	��"�/�2�1�/�&�"��	��������������-�:�=�F�J�G�F�:�-�!���!�)�-�-�-�-�-�-àìù����������ùìàÓÊÓÓÜàààà�����ʾ����������־Ͼž��������������f�r��������ͼԼӼʼ�����`�T�N�M�R�Y�f�����������������������������������������B�N�[�e�g�p�k�g�_�[�N�B�@�;�B�B�B�B�B�B�y�����������������������������{�y�w�s�y�ѿҿ׿ҿѿĿ��ÿĿпѿѿѿѿѿѿѿѿѿѹ���������������ܹ׹ܹ�����������������������
����������ùôñùÿ���a�n�z�Á�z�s�n�a�]�[�]�a�a�a�a�a�a�a�a����������������������y�x�x������������<�<�E�B�>�<�/�%�%�/�/�;�<�<�<�<�<�<�<�<���������������ּӼּ�������6�C�O�W�\�h�j�t�u�x�u�h�\�W�O�<�6�1�2�6ƎƳ��������������ƳƧƚƊƁ�}�x�w�zƁƎ���������������������������������������������!�3�D�F�:�!�������ߺ޺ߺ����@�3�'�'�"� �#�'�1�3�@�A�@�@�@�@�@�@�@�@E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�������������
��
���������������������ŔŠŭŶŹ����������ŹŭŠŝŗŔŒőŔŔ�������������������r�k�f�]�f�j�r�������E7ECEPE\E]E\EXEPECE>E7E3E7E7E7E7E7E7E7E7 N h 9 R % ? = # G Z B C " U X @ n ? 2 I Q 0   ? o 8 ! # } z S c 7 I * 9 q 8 l N 7 | n H : ( o B Y  2  h S M E l 2    \    8  u  �  �  �  �  �  �  #  �  �  �  s  e  �  K  �  <  �  F  �    �  �  �  3  �  �  �  �  i     �  �  �  1  O  g  �  t  Z  �  �  �  �  �  �  C  �  �  �  A  {  l  W  �  n  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  T  ]  e  m  u  }  �  �  �  �  �  �  �  �  �  w  n  b  U  H  *  )  )  (  +  <  L  \  m  ~  �  �  �  �  �  �  �  �  �  �  i  k  m  n  p  s  u  x  z  }  }  {  y  u  m  e  ^  T  K  B  �  �  �  }  w  p  j  c  ]  W  P  J  C  >  :  6  2  .  *  &  �    '  B  N  H  -  �  �  }  ?    �  u  F  5    �    h          
      �  �  �  �  �  �  �  �  �  �  �  �  �  �  �    Q  �  �  �  	  	!  	(  	)  	&  	  �  �  *  Z  B  �  �  D  v  �  �  �  �  �  �  �  �  �  i  6  �  �  �  -  �    O  �  }  �    ~  �  	*  	P  	^  	Z  	?  	  �  n  �  Q  �  �  �  I        %  "        	    �  �  �  �  �  �  k  F      �          �  �  �  �  �  �  �  �  r  S  0    �  �  �  z  I  ]  o  y  |  w  j  S  9    �  �  �  z  Q    �  w  �   �  _  m  u  x  {  ~    w  b  F  '    �  �  �  �  u  (  �  &  r  b  R  q  �  �  i  I  *  	  �  �  �  �  s  W  6    g  �  �  �  �  �  �  �  �  �  �  �  �  �  �  o  M  +    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  {  r  g  ]  R  G  <  0  (  *  -  /  0  .  ,  *  #      �  �  �  �  �  u  T  4    !  I  Y  i  u  z  w  m  ]  D  #  �  �  �  s  8  �  �  �  f  f  �  �  �  �  �  	  �  �  �  �  e    �  A  �    �  3  x  _  �  �  �  �  �  �  }  /    Q  Y  W  6  �  �  >  �  g  �  B  I  G  ;  *      �  �  �  �  �  �  �  �  �  �  �  {   �  k  �  2  k  �  �  �  �  w  G  �  �  B  �  =  �  �    <    v  �  �    :  S  e  o  g  Y  D  "  �  �  k  �  =  �  �  n  �  {  �  ;  �  �  2  �  �  �  �  �  c  �  0  [  <    �  e  �  �  �  �  �  {  r  g  Z  M  ?  /    	  �  �  �  v  C    �  �  �  �  �  �  �  �  �  v  b  N  9  "    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  n  Q  +    �  �  �  \  J  8  �  �  �  �  �  w  j  Z  D  )    �  �  �  �  m  F    �  H  x  �  �  �  �  �  �  �  �  �  �  �  �  g  I  &    �  �  �  �  �  y  o  d  Y  L  ?  2  %           �   �   �   �   �   �  "           �  �  �  �  �  �  �  �  �  �  �  �  �    9  [  Y  V  S  P  M  J  G  E  D  D  C  @  ;  6  2  �  �  c    O  >  -      �  �  �  �  �  �  �  z  g  S  <  #  
   �   �  P  2    �  �  �  �  o  W  4    �  �  `  P  �  �  �    :    �  �  �  �  �  �  �  �  �  n  U  9    �  �  d    �  �  �  �  �  �  �  �  �  �  �  �  �  �  f  1  �  �  y  6  �  �  0  <  Q  W  W  O  A  1    �  �  �  ^    �  M  �  �  �  2  V  B  =  <  /    �  �  �  �  �  �  �  |  N    �  y  3  �  !  �  �  U  �  *  '    �  �  >    �  A  e  8  
�  	G  P  �  3  /  +  '  $                 $  -  6  ?  H  P  Y  b  M  J  F  B  <  5  .  &      
    �  �  �  �  �  K  �  d  �  �  �  �  �  �  �  �  q  X  =    �  �  �  �  �  �  �  G  )  3  <  F  P  Z  d  n  x  �  �  �  �    )  N  r  �  �  �  �  �  �  �  �  �  �  �  r  b  R  B  0      �  �  �  �  ~  I  K    ,  8  >  :  %    �  �  v  �  y  &  �  =  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �     !  �  �  �  �  �  �  �  �  u  X  ;    �  �  �  ^  )  �  �  �  �  �  �  �  �  �  �  �  �  �  ~  �  �  �  �  �  �  �  �  �  2            �  �  �  �  �  �  �  �  �  �  �  y  e  R  4  ,      �  �  �  �  �  s  Q  ,    �  �  �  h  -  �  �  B  V  `  Z  M  3    �  �  �  Z    �  a  �  �    o  �  4  �  �  �  �  �  �  �  �  �  �  n  W  8    �  H  �  �    �  L  �  �  �  �  �    �  �  �  �  �  M  �  �  �  &  M  �  "  �  �  �  �  �  �  �  �  �  z  p  f    �  �    �  g  #  �  �  t  a  P  D  8  (      �  �  �  v  L  "  �  �  �  �  p      �  �  �  �  �  �  |  p  a  O  :  !    �  �  P  �  "  �  �  �  �  �  �  e  C    �  �  �  e  2    �  �    �   �  `  W  F  .    �  �  �  J  
  �  �  S  
  �  �    �  u  !  <    �  �  �  �  �  l  <  �  �  y  D    �  �  `  #  �  �