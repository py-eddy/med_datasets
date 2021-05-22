CDF       
      obs    8   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?pbM���   max       ?����n�      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M�oj   max       P��
      �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ��o   max       >+      �  l   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?�\(�   max       @E������     �   L   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ����P    max       @vc�
=p�     �  )   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @'         max       @N�           p  1�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�E        max       @��          �  2<   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       <49X   max       >hr�      �  3   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       B��   max       B,��      �  3�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       B�:   max       B,�      �  4�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >9g�   max       C���      �  5�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >B>[   max       C��      �  6�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          �      �  7|   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          E      �  8\   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          5      �  9<   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M�oj   max       Pa��      �  :   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�
=p��   max       ?�X�e,      �  :�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ;D��   max       >
=q      �  ;�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @@�Q�   max       @E������     �  <�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ����P    max       @vc�
=p�     �  E|   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @'         max       @N�           p  N<   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�E        max       @�_�          �  N�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         B�   max         B�      �  O�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?���,<�   max       ?��+J     �  Pl         
         
   7                     ]         ,      
         
            #   (         `   %   
   v                  �      !            4            
      
   7   o         O�F�OO�NO��N}(�N&�O�i4PXI�N<+�N�O�27O�cN���M�ojP��
N�<Nd�cP/�N��wNFsO��N� �N|OxN��O�t�O���Pa��N���Oņ:P_2�P�~N�GPd�UN���O n�O3�DNKA+O:��Pm��N�ՂOb��N[QNw�hN*�fP'fQOw��NsN�pTN��
N4��N�}FO{�Oo�Nm�N���N`�b��o�D��;D��;�o;ě�<#�
<#�
<49X<49X<49X<49X<e`B<e`B<�o<�C�<�t�<�9X<�j<���<���<���<���<�/<�/<�h<�<��<��=+=�P=�P=�w=#�
=#�
=#�
=,1=0 �=8Q�=P�`=aG�=e`B=e`B=m�h=q��=q��=�\)=Ƨ�=Ƨ�=Ƨ�=��`=�l�=�h=�h=�F>J>+)5BN[gpf^`XNB5)XOS[bhtv��������th[X~x{�������~~~~~~~~~~����������������������������������������jlp|��������������tj$"5B_t���������[E6$ �	BBINNZ[\glmgg[NBBBBB485BJaedjlk][OB:9774���������������������������
���������	

��������������)BK[bdbdkg[��/55<BNONMMIB75//////��������������������FNQORbt����������tVF)*5:95)|�����������||||||||����������������������������������������������������������������
+-/2//#
�������������������������������������������������
������Q]g������!������[Q*,6BO[[[UOB6********.,/4<H]apf\ZUH<0488.������*7=QQNB7���=9BNg�������xtgf`YE=_]hhpt�������th____}|�����
������}�����
#&#"
���35:<=DHUZacebaXUH<33lnswx{�����������znl�����  ��������������������������������� �������������������������)/<BHLNEB<6)56=BFMOPOKEBA;665555HIMU]bnv{�{nbaUIHHHH�������������������������)*( ���������������������������#"#/<=@?</*#########vrqpqwzz�������zvvvv��������������������a\UYahmnnnnaaaaaaaaa��������#" 
���������
##��������

	������"#''$#�������

����������������������������G�`�m���������������y�m�T�;�.�'�#�(�3�G�����������¼ɼʼмʼ��������������������@�L�Y�]�c�Y�L�@�9�:�@�@�@�@�@�@�@�@�@�@�)�6�;�6�,�5�)�������"�)�)�)�)�)�)�B�O�Q�V�R�O�I�B�9�B�B�B�B�B�B�B�B�B�B�B�����������������������{�y�s�q�k�u�w���������¾��������������s�Z�N�K�N�Z�f�������������ùö÷ù���������������������b�b�n�{�|ŇňŇŁ�{�n�b�b�[�W�a�b�b�b�b�'�@�Y�e�M�4�'�����ܻû����Ļл����'��������&�!����
������ݻ����!�-�:�>�F�Q�H�F�:�-�!���	������@�M�T�Y�_�Y�M�@�?�<�@�@�@�@�@�@�@�@�@�@��������.�6�S�0������Ǝ�\�J�6�_�hơ�������������������������������������������'�3�5�3�2�1�'����� �'�'�'�'�'�'�'�'�����"�/�;�L�Q�H�;�/�	����������������f�s�x�v�t�s�n�f�Z�U�Q�W�Z�b�f�f�f�f�f�f�<�B�H�I�H�A�<�/�'�'�/�8�<�<�<�<�<�<�<�<àìùþ����������ùìáàÚÓÉÌÓÝà�A�M�Z�f�p�q�h�g�f�e�Z�M�A�@�<�=�A�A�A�A�:�F�S�_�`�_�Y�S�J�F�:�6�4�4�:�:�:�:�:�:�m�y�����������y�m�e�`�U�T�N�S�T�\�`�l�m�����ùϹӹܹ���ݹܹϹù��������������s�����������������n�g�N�5����%�6�N�s�����ʼмּڼڼ߼ּ���������v�w���������5�A�R�\�Q�A�:�=�5�(������������ݿ����������ݿٿտԿݿݿݿݿݿݿݿ����������������������g�N�F�>�;�A�Z�g�����<�U�b�~�x�b�I�0���Ŀĳħĳľ�������<�O�tčęĈĀăĀ�h�[�J�6�������)�6�B�O�-�:�F�G�S�X�[�S�H�F�:�:�.�-�)�*�-�-�-�-Óàãÿ�����������à�z�e�`�a�g�nÌÈÓ����������������������������������������)�.�/�)����� ������������������)�6�9�6�+�&�����������������/�5�<�@�B�<�/�#�"��#�&�/�/�/�/�/�/�/�/�H�T�a�m�z���������r�m�a�T�H�B�;�2�2�;�H�@�M�r�����ּ����ͼ��������h�M�@�4�@�������Ľнݽ��ݽнĽ������������������������������ɺպɺ��������~�t�l�l�r�{����������������غ����������������	�
�������������������A�N�Z�\�`�Z�N�A�@�7�A�A�A�A�A�A�A�A�A�A��������&�/�1�+��	���������|�m�����������������ĽʽŽĽ�����������������������ǔǡǣǭǣǡǔǈ�}�}ǈǔǔǔǔǔǔǔǔǔ�F�S�_�l�x�������y�x�l�_�S�F�:�>�F�F�F�F�������������������������x�s�x���������������ûл׻лû�������������������������¦¦ª¦�m�b�U�L�R�\�d�n�{ŇŔŞŤšřŔŐŇ�{�mD�D�D�D�D�D�D�D�D�D�D�D�D�D{DwDrDtD{D�D��*�6�C�O�S�O�O�C�6�*�&��*�*�*�*�*�*�*�*EuE�E�E�E�E�E�E�E�E�E�EuErEqEmEiEuEuEuEuE7ECEPEZE\E`E\EPECE<E7E0E7E7E7E7E7E7E7E7 % F 0 R 4 6 [ N E n ) + J J E r 9 [ Q 3 1 = B V T ' i ] l e U J B  1 < ( B T h T X { E Z h F Q q T  B 1 H + K  �  d  ^  �  /  R  �  g  �    !      �  �  �  $  �  }  F  &  �  L  �  �  P  �  �  Z  �  �  �      "  �  Y  �  �  b  �  �  �  D  u  C  q    �  \  �    �    #  �<�9X<�t�<D��<49X<49X<���=��<�`B<�=�w<ě�<�/<�C�=�/<��
<�j=�+<�`B=\)=P�`='�=\)=L��='�=T��=�o=�\)=�P=D��>1'=��=H�9>!��=e`B=e`B=e`B=P�`=�hs>B�\=�C�=�9X=�O�=�\)=�+=�S�=� �=�G�=�`B=��=�l�=��#>.{>hr�>+>��>�B�~B>BBBZ,B��B4�B�B�B�)B�eB!��Bc�B#�eB�B�BB!&�B
"�Bw|BF0B!�[B��B! zB,KB�MB��B#*ABs�B24B�.B�\B	ʁBN�B�|B�HB}EB�B�BE�B��B"��Bz�B8{B'ǅB��BJyB+��B`2BrlB,��BO�B�BU�B�RBi�B�BHBHzBA�B��BH9B��BAB>IB<UB�mB��B!�B@KB#˟B��B�mB!=�B
?�BfLBKtB"2;B��B!5�B:$B�-B�:B"�Bs2BT@B:LB�iB	=�B^�B��B��B��B#�B�aB@B��B"�B@B?�B'��B�B�B+8�B@OB�$B,�BA�B��B<B�B%B<EBC_Ag��@�/?ʴ�A�K@A��WAG��AD��A��A�p0@�\�@��r@rJi@��!B�A� ?�ZA�9�A@��A��A�&-A>�@���Ak^�>9g�A�t�@�^ZA�>A~�A�^�A�uAڝ@~neA�B�A�>A��&A�koA�A�)�@�&A$�a@s.@R��@NQ�A�j�A�d�A!��B?�@���@��"@�D A��A�IC��B g�C���C���Ag.�@�r�?ȃ�AՀTA���AG/AD��A·�A��@��@�UI@m0�@Մ�BK^A�D�?���A���AA9A�A�_�A>�4@���Ak�V>B>[A�cO@�%�A�~�A~I\A��&A�Aڑ[@��A�j�A�bUA�R�AԆ<A�~A���@��A%  @-%@T$�@S��A�y�A�iA"�CBB@�HU@�m@�*A�9A�C���B v�C��C���         
         
   8                     ]         -               
            #   (         a   %      w            	      �      !            5            
      
   7   p                              1         '            E         )                        '      5      '   7   1      1                  ;                  5                                                                           5                                 '      5      '   '   -                        !                  1                                 O�ϥN�R�NO��N}(�N&�O�i4O�#�N<+�N�Of��O�cN���M�ojP[QN�<Nd�cO���Nv�NFsO��N�++N|N�{�N��#O�t�Ok#�Pa��N���Oņ:P
�P
�jN�GO�ÇN�%!N�qYO3�DNKA+O:��OӢ
N�ՂOb��N[QNw�hN*�fP!�GOw��NsN�pTN��
N4��N�}FO{�O/�uNm�N���N`�b  �  �  �    ^  �  )  �  �  W  �  �  �  �    �  <  J  �  �  .  d  �  '  �  d  �  �  �  
�  k  !  �  #  
  �  f  9  �  �    �  5    �  �  X  T  �  �  �  �  �  &  	)  J;D��;D��;D��;�o;ě�<#�
=��<49X<49X<���<49X<e`B<e`B=0 �<�C�<�t�=�w<ě�<���<���<�/<���<�<�h<�h=t�<��<��=+=�%=#�
=�w=�9X='�=0 �=,1=0 �=8Q�=��=aG�=e`B=e`B=m�h=q��=u=�\)=Ƨ�=Ƨ�=Ƨ�=��`=�l�=�h>
=q=�F>J>+)5BN_[WXXQNB65)Z[_hit�������ytoh[ZZ~x{�������~~~~~~~~~~����������������������������������������jlp|��������������tjXV\ht�����������ti[X �	BBINNZ[\glmgg[NBBBBB@=<=<=?BO[b_dede[OB@���������������������������
���������	

��������������)BNU[YX\[N3'�/55<BNONMMIB75//////��������������������Z`cgt�����������tf_Z)5775)|�����������||||||||���������������������������������������������������������������
#)+-$#!
���������������������������������������������������
������Q]g������!������[Q*,6BO[[[UOB6********.,/4<H]apf\ZUH<0488.����)4>?CE=5)��<;BNg��������{tg[IB<_]hhpt�������th____�������������������������
"!
�������<<>AGHUWaaca^UH<<<<<lnswx{�����������znl�����  ���������������������������������������������������������������)/<BHLNEB<6)56=BFMOPOKEBA;665555HIMU]bnv{�{nbaUIHHHH�������������������������)*(
���������������������������#"#/<=@?</*#########vrqpqwzz�������zvvvv��������������������a\UYahmnnnnaaaaaaaaa��������#" 
���������
##���������

�������"#''$#�������

����������������������������`�m�o�x�����y�i�T�G�;�.�,�(�.�/�:�G�`�������������ü��������������������������@�L�Y�]�c�Y�L�@�9�:�@�@�@�@�@�@�@�@�@�@�)�6�;�6�,�5�)�������"�)�)�)�)�)�)�B�O�Q�V�R�O�I�B�9�B�B�B�B�B�B�B�B�B�B�B�����������������������{�y�s�q�k�u�w��s�������������������s�f�a�`�^�a�f�p�s������������ùö÷ù���������������������b�b�n�{�|ŇňŇŁ�{�n�b�b�[�W�a�b�b�b�b�лܻ����'�4�@�<�7�'������ܻͻ˻ͻл�������&�!����
������ݻ����!�-�:�>�F�Q�H�F�:�-�!���	������@�M�T�Y�_�Y�M�@�?�<�@�@�@�@�@�@�@�@�@�@������������������ƚ�u�a�\�h�uƎƳ�������������������������������������������'�3�5�3�2�1�'����� �'�'�'�'�'�'�'�'������%�.�0�'�"��	��������������������f�s�u�s�r�p�j�f�Z�W�S�Y�Z�d�f�f�f�f�f�f�<�B�H�I�H�A�<�/�'�'�/�8�<�<�<�<�<�<�<�<àìùþ����������ùìáàÚÓÉÌÓÝà�A�M�Z�f�o�p�f�e�Z�M�A�A�=�>�A�A�A�A�A�A�:�F�S�_�`�_�Y�S�J�F�:�6�4�4�:�:�:�:�:�:�y������������y�m�`�T�S�T�V�^�`�m�s�y�y�����ù̹Ϲܹ�ܹعϹù¹����������������s�����������������n�g�N�5����%�6�N�s�������ʼӼռԼʼ�����������{�|����������5�A�R�\�Q�A�:�=�5�(������������ݿ����������ݿٿտԿݿݿݿݿݿݿݿ����������������������g�N�F�>�;�A�Z�g�����0�<�L�S�Q�E�:�#����������������������0�[�tĈĕĆ�}ā�}�h�[�L�6�!��	��7�B�J�[�-�:�F�G�S�X�[�S�H�F�:�:�.�-�)�*�-�-�-�-Óàìù����������������ìÇ�z�u�t�zÇÓ����������
������������������������������)�*�,�)��������������������)�6�9�6�+�&�����������������/�5�<�@�B�<�/�#�"��#�&�/�/�/�/�/�/�/�/�H�T�a�m�z���������r�m�a�T�H�B�;�2�2�;�H�����ʼԼ޼��߼ռ���������������������������Ľнݽ��ݽнĽ������������������������������ɺպɺ��������~�t�l�l�r�{����������������غ����������������	�
�������������������A�N�Z�\�`�Z�N�A�@�7�A�A�A�A�A�A�A�A�A�A��������&�.�1�*��	���������}�u�����������������ĽʽŽĽ�����������������������ǔǡǣǭǣǡǔǈ�}�}ǈǔǔǔǔǔǔǔǔǔ�F�S�_�l�x�������y�x�l�_�S�F�:�>�F�F�F�F�������������������������x�s�x���������������ûл׻лû�������������������������¦¦ª¦�m�b�U�L�R�\�d�n�{ŇŔŞŤšřŔŐŇ�{�mD�D�D�D�D�D�D�D�D�D�D�D�D�D|D{DwDzD{D�D��*�6�C�O�S�O�O�C�6�*�&��*�*�*�*�*�*�*�*EuE�E�E�E�E�E�E�E�E�E�EuErEqEmEiEuEuEuEuE7ECEPEZE\E`E\EPECE<E7E0E7E7E7E7E7E7E7E7 & C 0 R 4 6 Q N E g ) + J J E r 3 b Q 3 & = 8 Z T & i ] l U O J .  % < ( B # h T X { E Y h F Q q T  B  H + K    �  ^  �  /  R  -  g  �    !      �  �  �  S  �  }  F  �  �    �  �  �  �  �  Z  �  �  �  �    �  �  Y  �  �  b  �  �  �  D  d  C  q    �  \  �    w    #  �  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  -  L  d  r  {  �  �  x  e  F    �  �  l    �    �  r    �  �  �  �  �  �  �  �  �  �  �  �  e  E  '    �  �  �  �  �  �  �  �  �  �  �  �  }  o  `  M  6    �  �  ^  �  �  x    	        �  �  �  �  w  k  ^  M  <  $    �  �  �  y  ^  e  l  r  w  u  s  p  m  j  f  c  ^  Y  S  N  G  @  8  1  �  �  �  �  �  �  �  �  �  �  x  f  Z  O  E  ;  2  &     �  x  �  �  �  �  �        !  #  	  �  �  X  �  �  �  �   �  �  �  �  �  �  z  W  3    �  �  �  Q  
  �  u  +  �  �  P  �  x  W  4    �  �  �  c  =    �  M  �  q  �  �    �    B  N  P  E  3  7  U  R  K  A  1    �  �  �    �  4  �  �  �  �  �  �  �  }  k  X  A  (    �  �  �  �  �  d  +  �  �  �  �  �  �  �  �  x  i  Y  E  -    �  �  }  :  �  �  _    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  y  p  h  J  �  �  �  �  �  �  �  t  N  H  .  �  �  d  �  I  f  +  �          �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  ~  m  [  G  3      �  �  �  �  �  �        +  :  6  #    �  �  }  ;  �  �  H  �  F  B  D  F  H  I  E  A  =  5  '      �  �  �  �  �  �  �  �  �      �  �  �  �  �  �    j  U  =  $    �  �  �  �  �  �  �  �  �  �  �  �  k  @    �  �  V    �  �  M  7  w  �    '  +  %        �  �  �  �  �  �  �  i  H  )  	  �  �  d  P  <  )      �  �  �  �  �  q  U  ;  %    .  A  E  I  �  �  �  �  �  �  �  �  �  c  2  �  �  T  �  �  <  �  p  X  �  �     &  %    �  �  �  �  �  ~  B     �  |  �  �  F  �  �  �  z  f  N  1    �  �  �  q  M  )    �  �  =  �    Y  �  "  S  b  b  Y  E  -    �  �  �  t  #  �  ~  E  �  [  {  �  �  �  �  x  U  �  �  �  �  k  D    �  �  e  	  �  3  �  �  �  �  �  �  �  �  �  u  c  R  @  .    
  �  �  �  �  �  �  {  q  e  W  C  -    �  �  �  �  �  o  K  T  K    �  �  	�  
.  
q  
�  
�  
�  
�  
�  
~  
>  	�  	�  	I  �  :  �  �  �  �  �  J  G  k  e  T  >     �  �  {  %  �  T  �  �    �  �  "  \  !         �  �  �  �  �  �  �  t  `  M  5    �  �  �  M  	�  
�  i  �  `  �  �  �  �  �  �  2  �    
_  	�  }  +    �    "            �  �  �  �  �  �  j  7  �  �  �  ;    �  �  �  �        �  �  �  �  t  O  $  
  �  �  l  8  !  �  �  �  �  �  q  L  %    �        �  �  �  �  A  �    f  W  I  6        �  �  �  �  �  e  @    �  �  �  |  S  9  5  (    �  �  �  �  �    ]  5    �  T  �  {  �  E  u  
�  �  V  �     >  w  �  �  U  �  Y  �  
�  
1  	g  �  B  n    �  r  _  P  G  <  -    �  �  �  z  M    �  �  �  S   �   �      �  �  �  c  1  �  �  �  �  T    �  �  d    �  [  �  �  �  �  �  �  �  �  W  ,  �  �  �  c  &  �  �  E  �  �  4  5    �  �  �  �  �  Q    �  �  �  }  E    �  y  /  �  *      �  �  �  �  �  �  �  |  f  O  9  #    �  �  �  i  (  r  �  h  ^  K    �  �  �  [     �  �  X  �  b  �  :  w   �  �  x  d  L  9  /  %      "  %      �  �  �  i    �    X  >  #    �  �  �  �  ^  >    �  �  �  x  L  !      �  T  4    �  �  �  ^  3    �  �  �    j  Q  J  @  /    �  �  p  \  ?  !  �  �  �  x  Y  A  .      �  �  �  �  o  <  �  }  ]  7    �  �  �  ]  /  �  �  �  l  8    �  �  e  2  �  ~  h  O  5    �  �  �  �  Y  7      �  �  �  �  �  �  �  �  ]    �  �  J    
�  
�  
A  	�  	}  	  k  �  �    a  l  �  �  �  �  �  �  �  ,  �  K  �  :  �  �  �    �  �  �  h  &  �  �  �  �  o  Z  <    �  �  �  v  E    �  �  a    �  	)  	  �  �  �  S  !  �  �  h    �  X  �  �  0  �  ;  �  �  J  =    �  �  �  �  ^  4    �  �  �  4  �  d  �  �  %  �