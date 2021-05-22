CDF       
      obs    6   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?�n��O�      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M�WP   max       Pe��      �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       �o   max       =�{      �  \   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?z�G�{   max       @E������     p   4   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�z�G�     max       @vyG�z�     p  (�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @&         max       @P`           l  1   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @��          �  1�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ����   max       >F��      �  2X   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A��   max       B,W�      �  30   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A��`   max       B,&�      �  4   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       ?#�    max       C��}      �  4�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       ?1O�   max       C���      �  5�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          �      �  6�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          ;      �  7h   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          ;      �  8@   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M�WP   max       Pe��      �  9   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��2�W��   max       ?�H��      �  9�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       �o   max       =��#      �  :�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?z�G�{   max       @E������     p  ;�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�Q��    max       @vyG�z�     p  D   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @#         max       @P`           l  L�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @��          �  L�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         >�   max         >�      �  M�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�	� �   max       ?�H��     �  N�               *         N         <                     M      
                        !         
      �                  D   S   H      
   
         "                     Na�Nf|�NW�bN;�@O`�WO1p�N[�,P4zjO ��N`\'Pe��NA�N/�zO82lN ��Nm�N,0�PWǖO�vN�,N���N��O���O4qDO���NA%;Nv��O��JO��zM�WPN��HN�q�P-NLQ�N
�'ONKN��N��1O��P8P��OG��O<MUO)MO-)N���OJX*O�N��N��3O*R�N�>�N�IN�hE�o���
�o��`B�o:�o;�o;��
;ě�<49X<D��<�o<�C�<�C�<���<��
<��
<��
<�9X<�j<���<�/<�`B<�<��=o=+=��='�=0 �=49X=8Q�=8Q�=<j=@�=D��=H�9=H�9=L��=L��=L��=P�`=P�`=P�`=aG�=aG�=m�h=q��=�C�=�C�=�O�=��=�{=�{����������������������������������������rwz��������zrrrrrrrr1-.6BCMGB61111111111!/<HSTXXUQH</*#tz�����������������t���������������������������	��������)5BNXZTONB51)������������������������
/anvzUD</
�����������������������������������������������������������������������������������	��������������� �����������)BV_hjwvh[B6#/<IX_bgeaVH</*�������������hnnuz��������zvnhhhh��������������������s�����������������zsZWRPS[dhht}����{th[Z"&.0=IU^_\WIE<0(��������������������pmmotz�������tpppppp�����������������������)5=AA?;5)���d`dgktvttgdddddddddd��������������������LNSTahmnmiaTLLLLLLLL`\\at�����������tnj`��������������������`\aahmusmlaa````````������
#-*#
���3/0559BJNONJB5333333rqwz��������|zrrrrrrnz{��������������zzn����)6S[WKJNC)������1BLNNHB?6)�� !"%)15>@ACDDB@5) ����������������������#)/6:??6)�����').-+)��������������������������������

��������������������������������� ��������lmnsz������znnllllll���������������������" ������1+35BIDB851111111111LNQY[dgt|tpkg^[NLLLL�����������������������������������������0�=�I�V�\�V�I�H�=�2�0�/�$�#�$�(�0�0�0�0FFFF$F&F$FFE�E�E�F FFFFFFFF�zÇÓÚÔÓÇ�z�y�r�z�z�z�z�z�z�z�z�z�z��������$�%�(���������������������f�i�s���������s�f�_�Z�M�F�F�M�X�Z�Y�f�����ĿĿĿ������������������������������(�A�M�\�`�_�V�M�A�4����ݽȽн���(�T�`�m�t�u�t�t�v�m�`�T�G�G�F�F�G�K�T�T�T����������������������������������������;�T�y�������i�J�G�;�"���ҾǾ�
���;�������Ǽʼͼʼʼɼ������������������������������������������������������������ҹùϹܹ��������������ܹӹѹȹ��û�����!�+�!��������������*�.�5�*�*���������������s�������������������~�s�q�s�s�s�s�s�s���׾�������׾����f�O�K�Z�g�������������������������������������z�|�����������ĿĿʿǿĿÿ��������������������������a�l�m�z�|�{�z�q�m�a�W�T�P�R�T�Z�a�a�a�a�uƁƅƎƐƎƁ�u�o�j�u�u�u�u�u�u�u�u�u�u����������������������s�k�f�j�i�n�s�������������ȼʼʼʼü��������������x�}�������������Ƽ�������������i�f�`�[�f���������������ݻܻ���������h�uƁƎƖƎƂƁ�u�h�\�[�\�e�h�h�h�h�h�h����'�*�1�-�'�#�����޹Ϲ����ùϹٹ�f�s�w�y����������s�f�Z�M�A�?�;�>�C�M�f���������������������������������������̿T�`�m�y�{�}�}�y�m�b�`�T�P�I�T�T�T�T�T�T�0�<�I�Q�S�I�?�<�9�0�*�'�0�0�0�0�0�0�0�0�)�6�B�O�`�m�m�i�[�O�6���������������)�<�H�S�L�H�<�0�/�,�$�/�9�<�<�<�<�<�<�<�<�{ŇŔŔŠŦŠŔňŇ�{�z�{�{�{�{�{�{�{�{�N�[�g�z�j�g�_�[�N�I�B�5�3�8�D�N�n�{ŇŔŔŔœŇ�}�{�x�n�b�e�n�n�n�n�n�n������������������������������������������$�)�6�;�B�@�;�)����������������	��-�:�S�_�m�u��~�l�_�:�!������������-�~�������ºɺԺ������~�a�Y�S�M�U�Z�c�t�~�5�B�g�t�t�g�[�N�B�5�,�)�%�)�5��"�-�0�,�(�"���	������������� �	�����	��������	���������������������4�7�@�B�M�Y�]�Y�X�M�@�4�1�$�#�"�#�'�2�4��!�-�0�:�E�F�:�-�%�!���������E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�EEyEuE��M�O�Y�f�i�n�f�`�Y�R�M�@�5�7�6�9�@�D�M�MDbDoD{D�D�D�D�D�D�D�D�D{DoDkDbD_DbDbDbDb���������������������������������������������ĽȽȽɽǽĽ�����������������������������������	����������������������������(�5�9�A�5�(�'�������������N�T�Z�g�h�s�z�{�s�l�g�Z�W�N�J�B�N�N�N�N * V + 8  q ? / K ' O j G - p 1  A - N 3 H r ! " l \ 7 6 ? 7 2 / q y 2 9   , , $ o L G f a > ; j J P R Z �  0  �  k  E  �  �  u  (  v  ]    �  X  �  h  ~  �  �  \  %  �  =  E  �      �  �  ^    �  �  �  ^  `  �  �  �      F    �  c  �  �  �  0    �  �  �  L  ܼ��ͼD���o��o=�w<�t�;ě�=���<ě�<�o=�hs<���<�/=o<ě�<�j<�j=Ƨ�=8Q�=+=o<�h=Y�=e`B=u=#�
=��=�hs=�+=<j=Y�=H�9>F��=L��=H�9=��P=e`B=ix�=>+=��#=�7L=u=y�#=��
=�C�=�j=��=�-=��T=��
=�-=�9X=�E�B">cB�B��B��B��B�AB(�B"bBL�B��B^B"��B�B �ABZ�B��BQBgB�vB̖Bz8B�Be�B�B&;.B!�B
�B��B*B	gB �mA��B
�B�1A�@B�YB��B�>B7�B�OB�\B��B�1B̠B*B*�%B�BL:BɶBN4B,W�B�$BgCB	 �B"?�B�BI�B�iB��B�yB(HB"1�B>%B��B!�B"��B�^B AB��B��B=B"B�dB@BK8BF�B=0B�KB&;�B �!B	�VB<�BM�B	WB K>A��`B
��BQ}A��dB@"B��B�xB?�B��B8�B�B�B:eB�B*�B��BH�B?�B?,B,&�B�jBBB�@@�B
��C��}A��A�3YA?�;Au�jA5�;Ai@lA���Abص@���A�o)?#� @coA��AF��AM��A�$:AvQ|A�ӋB'ADΩ@�@�]@�E�BՁ?Ei�A@B�5Ai��A�x�A�7�AÒ�A��!A�~�A�6A�I�A�@�@{�V@�A�vBA��A�oF@�4@p��C�@ճ�C��
@�<�A#�1A�A���A��m@��B
?pC���AɁ�AҢ�AC�Av��A7�Ah��A�x�AaZ@���AІz?1O�@dp�A��mAGi_ANۏA�s�Au SA��0B�AGG@�C�@�4�@��QB��?N��A?�B��Ai�A���A�|�A�p�A��A�w@A���A��EA�mr@x�@�A�uA�u�A�@�$c@s�C��@��C��@�ґA#/�A�!A�zhA�~       	         *         O         <                     N                        	      "         
      �               	   D   S   I      
            #                                             /         ;                     3               !               !               +                     -   #                                                                        ;                                                                                       )                                          Na�N5� NW�bN;�@O	6O1lN[�,O��*O�MN`\'Pe��NA�N/�zO$6uN ��Nm�N,0�O�^OtH�N�,N���N��O��N�\�Oq��NA%;Nv��O��CN�hM�WPN��N�q�OIzjNLQ�N
�'O3��N��N��1ORYgP
�O���OG��O<MUO)MO�pNr�O7��O�N��N��3O*R�N��JN�IN�hE  1  ^  @  �  �  |  �  �  o  e  �  G  �  ,  �  x  �  5  �  L  f  [  8  j  *  5  l  u  ;  V  �  m    �      �  A  
v  	a  1  /  z     �  �    C  `  �  `  G  	  ?�o����o��`B<#�
;D��;�o=\)<o<49X<D��<�o<�C�<�t�<���<��
<��
=m�h<���<�j<���<�/<�=�P=+=o=+=#�
=Y�=0 �=8Q�=8Q�=��#=<j=@�=L��=H�9=H�9=�O�=y�#=��P=P�`=P�`=P�`=m�h=e`B=u=q��=�C�=�C�=�O�=��P=�{=�{����������������������������������������rwz��������zrrrrrrrr1-.6BCMGB61111111111!#//<=HKOQIH</'##{������������������{������������������������������������)5BNVWPNB5)������������������������
/anvzUD</
�����������������������������������������������������������������������������������	��������������� �����������#)6BOVY[]^][OB6#"!!&/<>HQUZ^a[UH</"�������������hnnuz��������zvnhhhh��������������������{�����������������WVX[_htu~���}wth][WW$(0<IS\]ZVIC<0)#��������������������pmmotz�������tpppppp����������������������
),4565+)��d`dgktvttgdddddddddd��������������������LNSTahmnmiaTLLLLLLLLrnot|�������������tr��������������������`\aahmusmlaa````````�����
#$#!
���3/0559BJNONJB5333333rqwz��������|zrrrrrr�����������������������)6OUVRBB6)��� ��)6=BDC>6)	  !"%)15>@ACDDB@5) ����������������������#)/6:??6)����#),,)��������������������������������

��������������������������������� ��������lmnsz������znnllllll�������������������� !      1+35BIDB851111111111LNQY[dgt|tpkg^[NLLLL�����������������������������������������0�=�I�L�I�E�=�0�0�0�$�$�$�)�0�0�0�0�0�0FFFF$F&F$FFE�E�E�F FFFFFFFF�zÇÓÚÔÓÇ�z�y�r�z�z�z�z�z�z�z�z�z�z�����������������������������f�s�~�����}�s�f�\�Z�M�K�J�M�R�[�\�\�f�����ĿĿĿ�������������������������������(�4�A�L�Q�O�I�A�4�(�������������`�m�q�r�r�r�t�m�`�T�P�H�H�M�T�X�`�`�`�`����������������������������������������;�T�y�������i�J�G�;�"���ҾǾ�
���;�������Ǽʼͼʼʼɼ������������������������������������������������������������ҹܹ����
����������ܹ׹ӹϹɹϹܻ�����!�+�!��������������*�.�5�*�*���������������s�������������������~�s�q�s�s�s�s�s�s�ʾ׾�����׾ʾ����������������������������������������������������������������ĿĿʿǿĿÿ��������������������������a�l�m�z�|�{�z�q�m�a�W�T�P�R�T�Z�a�a�a�a�uƁƅƎƐƎƁ�u�o�j�u�u�u�u�u�u�u�u�u�u�����������������������s�k�m�k�o�s�v������������¼����������������������������������������������������l�f�a�^�f�r���������������ݻܻ���������h�uƁƎƖƎƂƁ�u�h�\�[�\�e�h�h�h�h�h�h������)�/�*�'�������ܹϹùȹϹܹ�Z�f�m�s�u�w�v�s�f�`�Z�X�M�L�I�M�N�V�Z�Z���������������������������������������̿T�`�m�x�y�z�y�u�m�h�`�T�R�K�T�T�T�T�T�T�0�<�I�Q�S�I�?�<�9�0�*�'�0�0�0�0�0�0�0�0�)�6�B�I�O�Q�U�S�O�I�B�6�)������&�)�<�H�S�L�H�<�0�/�,�$�/�9�<�<�<�<�<�<�<�<�{ŇŔŔŠŦŠŔňŇ�{�z�{�{�{�{�{�{�{�{�N�[�g�x�t�i�g�[�[�W�N�B�5�5�9�F�N�n�{ŇŔŔŔœŇ�}�{�x�n�b�e�n�n�n�n�n�n��������������������������������������������)�6�6�/�)��������������������-�:�S�h�n�x�x�h�_�S�:�!������������-�~�������������������������r�c�^�e�i�r�~�5�B�g�t�t�g�[�N�B�5�,�)�%�)�5��"�-�0�,�(�"���	������������� �	�����	��������	���������������������4�?�@�K�M�W�Y�[�Y�T�M�@�4�'�'�%�%�'�3�4��!�-�/�:�D�B�:�-�&�!���������E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E{E}E��M�O�Y�f�i�n�f�`�Y�R�M�@�5�7�6�9�@�D�M�MDbDoD{D�D�D�D�D�D�D�D�D{DoDkDbD_DbDbDbDb���������������������������������������������ĽȽȽɽǽĽ����������������������������������������������������������������(�5�9�A�5�(�'�������������N�T�Z�g�h�s�z�{�s�l�g�Z�W�N�J�B�N�N�N�N * A + 8  r ?  L ' O j G " p 1  @ $ N 3 H l   l \ $ ; ? 2 2  q y 1 9   . -  o L G T _ ; ; j J P G Z �  0  f  k  E    �  u  j  V  ]    �  X  X  h  ~  �  ^  �  %  �  =  �  �  �    �  E      �  �  �  ^  `  ~  �  �  �  �      �  c  L  �  �  0    �  �  �  L  �  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  1  *  #      	    �  �  �  �  �  �  �  �  �  �  �  �  �    5  S  ]  ]  Z  Q  H  <  0      �  �  �  �  �  �  �  �  @  A  B  D  >  8  2  *  !        �  �  �  �  �  �  �  �  �  �  z  q  c  T  E  1      �  �  �  �  �  {  d  L  5    �    X  �  �  �  �  �  z  T    �  �  =  �  /  �  �  �  �  N  j  y  |  z  p  ]  J  @  Z  @  ?  2    �  �  )  �  /  �  �  �  �  �  �  �  �  �  v  k  e  c  b  `  _  ]  \  Z  X  W  �    �  #  Q  q  �  �  �  s  M  #  �  �  *  �  �    �  �  `  j  n  n  h  e  d  X  G  2    �  �  8  �  �  F  �  �  Z  e  \  T  K  C  <  5  .  (  "              D  t  �  �  �  �  �  �  �  �  j  Q  S  M  �  �  u  Z  �  �  �  #  Q  ,  G  F  E  D  D  C  B  A  A  A  A  A  @  ?  <  :  7  4  1  .  �  �  �  �  �  �  �  �  �  �  �  �  �  �  t  V  7    �  �    %  !    �  �  �  �  q  G    �  �  �  Y  b  I    �  �  �  �  �  �  �  �  r  `  L  4      �  �  �  �  �  �  �  �  x  p  h  _  W  O  G  :  +      �  �  �  �  �  �  �  m  W  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  O  �  �  9  }  �  �      3  2    �  �  i    �  �  �  _  �  �  �  �  �  �  �  �  z  a  G  -    �  �  �  ]  W  J    L  1    %  +      �  �  �  �  �  k  A    �  �  �  V  "  f  `  Z  R  F  :  *      �  �  �  �  �  x  I     �   �   �  [  Z  X  V  T  R  Q  O  M  K  E  9  .  "          �   �   �  )  6  6  &    �  �  �  �  �  �  i  =    �  a     �  '  �  J  S  ]  e  i  j  e  [  E  $    �  �  }  I    �  �  "   �    (  '       	  �  �  �  �  Y    �  �  ^  .  �  �  �  =  5  '    .  W  u  t  r  p  n  j  f  b  \  W  M  A    �  P  l  c  [  R  I  >  3  (        �  �  �  �  h  L  3       p  o  q  Z  <      �  �  �  �  p  L    �  �    �    [  +           %  ,  1  8  ;  5  &    �  �  �  T    �  �  V  K  @  5  *        �  �  �  �  �  �  �  w  ^  D  +    �  �  �  �  �  �  �  �  �  �  ~  r  c  K     �  �  �  R    m  b  V  K  @  4  '      �  �  �  �  �  �  �  �  �  �  �  �  ;  �  �  2  �  1  �  �  �    �  �  �      �  	�  �  �  �  �  �  �  �  x  p  g  ^  V  M  D  :  1  (    �  �  �  h    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  u  j        �  �  �  |  V  8      �  �  �  i    �  4  |  �  �  �  �  �  �  �  �  �  �  �  s  c  O  <  (    �  �  �  �  A  8  /  #      �  �  �  �  �  k  I    �  �  �  h  .   �  	(  	�  	�  
0  
\  
s  
o  
R  
  	�  	�  	T  	  �    O  u  �  �  �  	@  	[  	`  	^  	?  	  �  �  q  a  c  q  �  z  #  �  �  �  �  �  �  �    !  (  /  1  /  #    �  �  ~  1  �  !  o  �    �  /  '        �  �  �  �  �  �  t  P  ,    �  �  �  g  %  z  s  l  e  ^  U  L  @  3  #    �  �  �  �  v  c  K     �         �  �  �  �  �  �  x  a  E  "  �  �  �  �  �  �  �  q  }  �  �  �  k  O  2  
  �  �  `    �  y    �  h  $  @  �  �  �  �  �  �  �  �  �  �  u  W  5    �  �    I    �        �  �  �  �  �  �  k  0  �  q    �     ]  �  �  o  C  6  )      �  �  �  �  �  �  ]  0  �  �  �  a    �  5  `  O  =    �  �  �  �  [  8          �  �  �  _    �  �  P  &    �  �  �  S    �  �  o  *  �  p     �  �  A   M  `  L  7     	  �  �  �  �  �  �  n  Z  ?  !  �  �  �  �  �  5  B  B  6  !    �  �  �  �  |  *  �  X  �  l  �  o   �   c  	  �  �  �  �  �  �  �  �  �  �  r  d  S  ;  $     �   �   �  ?  2  $    
  �  �  �  �  �  �  �  �  �  �  o  R  5     �