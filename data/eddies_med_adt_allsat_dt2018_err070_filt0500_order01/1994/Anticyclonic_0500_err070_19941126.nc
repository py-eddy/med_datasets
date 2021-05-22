CDF       
      obs    4   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?�1&�x��      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M�	   max       P���      �  |   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ��hs   max       >V      �  L   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>��\)   max       @E�z�G�            effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       �ۅ�Q�    max       @vZ�Q�        (<   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @'         max       @O            h  0\   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�        max       @�ɠ          �  0�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �0 �   max       >��#      �  1�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A��   max       B-/      �  2d   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�~x   max       B,�L      �  34   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >�Ce   max       C��K      �  4   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >���   max       C��%      �  4�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max         {      �  5�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          7      �  6t   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          )      �  7D   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M�	   max       P
J�      �  8   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��t�k   max       ?�P��{��      �  8�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ��o   max       >!��      �  9�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�33333   max       @E�Q��        :�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��G�z�    max       @vZ�Q�        B�   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @"         max       @Q�           h  J�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�        max       @� �          �  K,   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         @   max         @      �  K�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?���҈�   max       ?�F
�L/�     @  L�     z               	   B         3                   0   +               (      &      y                              "      	            t   "   O               #      O�P���N�*�N�N,�nO&��M�	P0�NYQ!N�x�P��OR��O�c)P�N���O�2O���PZN �XO��O���OXO��M�"MO���N�.�PW�5M���O2��N�%DO�};N�@�N�N9R1NT܎N�_EO^��Oa�N
��N�IGO,�N���O��O��5OՊ�NŽ8N��SOX΂N4��O$�Nc�~N"K;��hs�u�ě����
�D����o;��
;�`B;�`B<D��<D��<T��<e`B<�C�<��
<ě�<���<���<���<�/<�/<�h<�<�<�<��=o=\)=�P=��=��=��=�w=#�
='�=0 �=8Q�=8Q�=<j=L��=L��=L��=T��=Y�=aG�=ix�=ix�=�O�=�\)=��=�9X>Ve_`fgt����������tgee%  #-B[t��������[N5%��������������������z|��������zzzzzzzzzz����������������������������������������AABGOPWOOBAAAAAAAAAA<;DBg����������s[NB<���	�����������������������������%!),6Ohm|~~�|th`X)%����������������������������������������Wg���������thtxotc[W
#%#
[]bt��������������t[���
#/<DHQOF>/#RVan�����������nUWVR)&),6BBDB:6)))))))))�����
#$## 
�������������������� ������������������������)**+)')5BNgv���ytgN?)7224<DHOUU\\UOH<7777������6BHB=.�������

�����������������������������������	""$"	����������#)3)&$���;=;@BNPWYWWNGB;;;;;;),-)���� �������������������������������������������������������~~����������������~��������������������$)686+)&}}����������}}}}}}}}�������������������)-2)(������

��������������' ������������������������������

��������!),/.*)%QPQRW[hqt|������th[Q������������������������
#++)$#
�������������������������������������������a�n�zÇÈÍËÉÇ�{�z�n�c�b�a�[�Y�_�a�a������)�B�Y�]�W�I�6�$��������ÿÿ���޻����������������������������������������f�s�w�t�t�s�f�f�e�b�f�f�f�f�f�f�f�f�f�f�����������������������O�C�B�6�*�������������*�6�;�C�O�O�ʼּ޼���ּʼʼƼʼʼʼʼʼʼʼʼʼ��5�N�������������������g�I�5������5������������������ùòóùù���������������������������������������������������Ҿ����ʾϾ׾�����׾ʾ�����c�S�Z�f�����a�n�y�w�~�{ÇàìñìÖÇ�z�n�a�Z�_�V�a�N�g�n�i�Z�5�(������ٿֿݿ����(�5�N�����������������u�n�c�W�a�f�m������������������������������������������������������������ƾʾ׾׾ʾ��������s�j�b�d�p�������(�2�>�F�G�@�5�*�(����������N�g�����}�s�p�g�Z�N�5������(�5�A�N�ݿ������������ݿۿݿݿݿݿݿݿݿݿݿm�y���������������{�y�m�l�`�T�G�I�T�`�m���	�"�.�G�T�\�k�c�G�;�.�"��	�������������
���#�*�/�%�#���
�����������������)�,�0�0�)���������üû�����������m�y�����y�m�`�^�`�j�m�m�m�m�m�m�m�m�m�m¦²º¼»¼²�t�g�b�_�m�t��������� ���������������������뻷�л�����������ܻ����x�b�x�����������������x�}�|������������"�/�;�B�H�N�I�E�>�;�2�"���	��	���H�T�U�a�i�m�n�m�m�a�^�T�I�H�G�F�H�H�H�H�#�0�<�I�U�c�f�a�U�D�0�(�#�������	��#�0�<�I�U�V�W�U�I�<�0�#��#�-�0�0�0�0�0�0�f�o�o�i�f�Z�W�X�Z�`�f�f�f�f�f�f�f�f�f�f���������������������������������������������������������������������������������������*�3�*�����������������������Ľнݽ�����ݽнĽ�����������������	���(�4�?�A�L�H�A�4�(�����������'�3�7�3�2�/�'�$�����'�'�'�'�'�'�'�'�a�m�z�|�����z�m�a�a�[�\�a�a�a�a�a�a�a�a�h�uƁƍƎƑƐƎƌƎƏƎƁ�u�m�h�^�U�X�hƁƎƙƚƢƞƚƎƂƁ�{�{ƁƁƁƁƁƁƁƁD�D�D�D�D�D�D�D�D�D�D�D�D�D|DtDvD�D�D�D��y���������������������y�v�l�j�`�Y�\�l�y�Ϲܹ���������ù������������������"�/�;�A�H�L�H�H�;�/�"������"�"�"�"�	���	�����������������������	�	�	�	�4�M�Y�f�r�w�v�r�p�f�Y�N�M�@�4�1�.�*�'�4���
������
�	�����������������������N�[�g�i�t�~�s�g�[�N�B�@�5�9�B�I�N����������������������������������������E7ECEPEUE\EgE\EPECE=E7E5E7E7E7E7E7E7E7E7 B ( R | : B D 7 { 7 7 [ q A = Y # 0 5 C a T O o E  , v . Q ( H j . < ] : Q J @ T O B O N  f ? n ' V `  X  U  �  �  S  �    *  �  �  �  �  �  �  �  P  }  u  C  ^  4  3  �    *  �  �  b  �  �  T  �  f  C  \    �  M  5  �  }  �  K  �  %  �  �  �  y  f  �  k�0 �>��#;o�D��;�o<u<e`B=��<u<�=}�=o=8Q�=�w<ě�='�=��=�7L<�`B=@�=H�9=49X=�O�=o=�C�=T��>��=�w=H�9=L��=�%=49X=,1=@�=T��=aG�=��w=��P=]/=�+=�7L=ix�>,1=�->	7L=�o=�%=�j=���=�/=���>�B	��B�wB"��BDB�rBm B�&B
FuBp�B�B�BH�B�BqB�B
��B��B�B�B�?BI|BډB!XB��BF�B�B��B#��BsA��B��B `B&]B �BB�dBҎB"�B(-B �B��B�B��B-/B��BJ�B��B�B��B-BGBB�B	��B�HB"��B˳B��B<B��B
�B��B��B�B�cB��B�BʑB�B�Ba�B�OB�zB8�B�aB§BIQB��BB��B#�aB��A�~xBI?B<�B�BB�B��B��B"@'B=8B �xB@PB+B?�B,�LB��BJ�B?5B�IB��B2B>�B�kA��A��y@���AA��A�0wA���A ��A��
A���A�hkAMC�A���A���A��!A��AI�A��$A���ADAk�A`�gA���AҪ�Ak�:A��IA�F�@�o#@�EA�=AA��A���A�OA@�AHr1A���A��hA'URA6$?��tA��LB�}B�CC��A��>�CeA��A��H@���A�O�A�|�@C��KAǄ^Aӈ�@�%AB�cA�{�A�M�A �A��+A�~AЀ�AM�Aȩ9A�R�A���A���AI
=A�~A��9A��Al��AbA���A�z�AlėA�z�A҉M@��@���A�.A�w]A�RA�m�AAAH�2A�[A��eA&�A6�M?���A��wB��B�C�� A��>���A�A�A�f�@��>A��A�q�@�E�C��%     {   	            
   C   	      3                   1   +               (      '      y                              "      	            u   #   P               $            3                  7         )   !   +   +      '      %                     !      1                                                      #                                                               )                                 !      !                                                                           N�1Oȶ.N�*�N�N,�nO&��M�	O��NYQ!NU�UO7��N���Ok~�P
J�N���O���O%��O�iHN �XOYO=��OXO���M�"MO���N�xOڭ�M���O2��NfDO���N�@�N�N9R1N1	sN�_EO?�	O 0N
��N�IGO,�N���O5�ZOX�O���NŽ8N��SOX΂N4��O��Nc�~N"K;  �  !X  �  >  -  �  �  r  "  �  �    7  y    S  �  :  �    /    �  �  �  Z  m  �  �  �  �    �  �  �  +  �    �  �  �  �  +  �  l  �  �  �  �  �  R  ۽�o>!���ě����
�D����o;��
=0 �;�`B<�o=#�
<�1<ě�<�t�<��
<�/=,1=#�
<���<�`B<��<�h=C�<�<�=+=�hs=\)=�P=�w='�=��=�w=#�
=0 �=0 �=D��=@�=<j=L��=L��=L��=���=u=�\)=ix�=ix�=�O�=�\)=���=�9X>Vcdgnt�������tkgcccc;757;CN[gt����t[NB;��������������������z|��������zzzzzzzzzz����������������������������������������AABGOPWOOBAAAAAAAAAAf_]`egt����������tgf���	�����������������������������==?BEO[fhjknonkh[OC=����������������������������������������X`g�����������zqvf\X
#%#
gbet��������������tg
##/8<=@=<3/#geegnz�����������zng)&),6BBDB:6)))))))))�����
###"
����������

���������� ������������������������)**+)')5BNgv���ytgN?)9336<GHJRUYZULH<9999������&)'"��������

��������������������������������	"#"	������������ ')#!����;=;@BNPWYWWNGB;;;;;;),-)���� �����������������������������������������������������������������������������������������������$)686+)&}}����������}}}}}}}}�������������������)-2)(���������

��������������	���������������������������������

��������!),/.*)%QPQRW[hqt|������th[Q������������������������
 #)*(#"
��������������������������������������������n�zÂÇÊÇÇÂ�z�n�j�a�`�]�a�e�n�n�n�n��������)�4�<�=�9�)���������������뻅���������������������������������������f�s�w�t�t�s�f�f�e�b�f�f�f�f�f�f�f�f�f�f�����������������������O�C�B�6�*�������������*�6�;�C�O�O�ʼּ޼���ּʼʼƼʼʼʼʼʼʼʼʼʼ��Z�g�s�������������������s�g�\�O�M�R�Z�Z������������������ùòóùù���������������������������������������������������Ҿ����ʾӾ׾ݾ޾׾Ӿʾ�������������������ÓàääàÓÏÇ�z�n�a�^�a�n�zÇÊÏÓÓ�5�A�N�Z�e�a�Z�N�5�1�(������(�(�4�5���������������������x�r�f�b�m��������������������������������������������������������������¾žɾ��������s�p�g�j�p�x�������(�-�5�5�/�(� �������������5�A�N�Z�c�k�e�a�\�N�A�5�(� �����(�5�ݿ������������ݿۿݿݿݿݿݿݿݿݿݿm�y���������������z�y�u�m�`�T�I�K�T�b�m�	�"�.�;�G�T�`�`�W�T�G�;�.�"��	�������	�����
���#�*�/�%�#���
����������������������)�-�.�)�"���������ÿ��������m�y�����y�m�`�^�`�j�m�m�m�m�m�m�m�m�m�m¦²º¼»¼²�t�g�b�_�m�t������������������������������뻷�ûܻ���������ܻû����������������������������x�}�|������������"�/�;�B�H�N�I�E�>�;�2�"���	��	���T�a�g�m�n�m�g�a�_�T�J�I�T�T�T�T�T�T�T�T�0�<�I�U�`�c�b�[�U�<�/�#�������#�0�0�<�I�U�V�W�U�I�<�0�#��#�-�0�0�0�0�0�0�f�o�o�i�f�Z�W�X�Z�`�f�f�f�f�f�f�f�f�f�f���������������������������������������������������������������������������������������*�3�*�����������������������Ľнݽ��������ݽнĽ�����������������(�4�=�A�K�F�A�4�(����� �����'�3�7�3�2�/�'�$�����'�'�'�'�'�'�'�'�a�m�z�|�����z�m�a�a�[�\�a�a�a�a�a�a�a�a�h�uƁƍƎƑƐƎƌƎƏƎƁ�u�m�h�^�U�X�hƁƎƙƚƢƞƚƎƂƁ�{�{ƁƁƁƁƁƁƁƁD�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D~D�D�D�D������������������������x�l�`�]�b�c�l�y���ܹ����	����Ϲ����������������ùϹ��"�/�;�A�H�L�H�H�;�/�"������"�"�"�"�	���	�����������������������	�	�	�	�4�M�Y�f�r�w�v�r�p�f�Y�N�M�@�4�1�.�*�'�4���
������
�	�����������������������N�[�e�g�t�|��t�q�g�[�N�B�B�7�:�B�L�N�N����������������������������������������E7ECEPEUE\EgE\EPECE=E7E5E7E7E7E7E7E7E7E7 1  R | : B D  { = : � Y < = =  5 5 B Z T O o E #  v . = * H j . = ] 8 N J @ T O . > K  f ? n   V `  �  �  �  �  S  �    �  �  q  �  �    {  �  u  ]  b  C  N  �  3  W    *  �  �  b  �  �    �  f  C  @    �  ,  5  �  }  �  x  �  ^  �  �  �  y  3  �  k  @  @  @  @  @  @  @  @  @  @  @  @  @  @  @  @  @  @  @  @  @  @  @  @  @  @  @  @  @  @  @  @  @  @  @  @  @  @  @  @  @  @  @  @  @  @  @  @  @  @  @  @  �  �  �  �  �  �  �  �  �    H  
  �  ~  5  �  �  )  �  �  �  �  �  �  �  �  �   �  !8  !K   �   V  u  >  {  !  �  _  =    �  �  �  �  z  h  T  @  +      �  �  �  �  �  �  �  �  �  >  <  :  7  5  3  0  .  ,  *  %            �  �  �  �  -  #          �  �  �  �  �  �  �  �  �  	    Q  �  �  �  �  �  �  �  z  f  L  1    �  �  �  f  8  
  �  �  7  �  �  �  �  �  �  �  �  �  �  �    b  B  "     �  �  �  i  A  
  c  �  �  �  �  "  I  j  i  k  k  T  !  �  q  �  +  "    "  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �        �  �  �  �  �  a  9    �  �    �  �    1  F  Z  n  |  �  �  �  �  d  #  �  
  U  �  �  �  �  �  �  �  �  �  �      �  U  2    �  �  k  �  �  �  �  �  �  �    0  7  ,        �  �  n  (  �  t    ~    _  w  h  W  F  ;  +  
  �  �  �  �  j  `  X  I  ;  7  ,                      �  �  �  �        �  �  �  �  r  :  6  ,  F  S  Q  H  ;  )    �  �  �  v  P  +  �  �  m  ^  �    E  t  �  �  �  �  �  �  �  f  ,  �  �  .  �  �  �  �  |  �  �  �    '  9  9  +    �  �  q    �  <  �     B  ]  �  �  �  �  �  �  �  �  �  {  v  q  l  h  c  ^  Y  U  P  K              �  �  �  �  �  _  +  �  �  ;  �  -  �        -  /  *      �  �  �  �  h  6     �  �  Q  �  �  5      �  �  �  �  �  w  R  +  �  �  �  Z    �  I  �  g   �  �  �  �  �  �  �  �  �    I     �  5  �  N  �  `  �  �   �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  r  Q  .    �  �  f    �  }  (  �  �  s  �  �  �    S  X  Y  V  O  D  7  %    �  �  �  �  �  �  h  M  2      
0  
j  
�  
�  0  \  l  g  T  5    
�  
R  	�  	-  c  n    �  C  �  �  �      
    �  �  �  �  �  �  d  ?  !    �  �  �  �  �  �  �  �  �  �  �  �  �  {  q  `  J  *    �  �  �  �  E  o  }  e  S  B  .       �  �  �  �  �  \    �  s    �  |  �  �  �  �  �  �  n  Q  3    �  �  �  �  �  3  �  r      �  �  �  �  �  �  �  w  a  X  \  `  _  W  O  F  7  )    �  �  �  �  �  {  q  f  [  O  C  7  +    	  �  �  �  �  �  �  �  �  �  �  �  z  o  d  X  M  @  ,      �  �  �  �  �  z  {  �  �  �  �  �  �  �  �  �  p  I  _  t  r  u  �  
  d  +      �  �  �  �  �  p  H    �  �  q  1  �  �  ]    �  �  �  �  �  �  �  r  `  J  0    �  �  v    �    n  �   �  �      �  �  �  �  �  T    �      �  1  �  B  �  W  .  �  �  �  �  �  �  �  �  �  �  �  �  u  h  \  N  A  3  $    �  �    l  ]  M  <  +    �  �  �  o  @  �  l  �  R  �  5  �  |  f  L  0         �  �  �  �  �  ^  #  �  �  L  �  �  �  �  �  �  �  l  W  A  +       �  �  �  �  �  \  L  D  ;  #  f  �    #  +  &    �  �  [  �  "  <  ;  %  �  �  
  �  z  �  �  �  �  �  �  ~  e  O  B  S  M  0  	  �  y  $  [  �  
�  ,  X  i  l  `  M  /    
�  
�  
]  
  	�  �  +  ^  �  �  �  �  �  �  �  �  �  �  u  g  V  D  0    �  �  �  �  v  L  !  �  �  x  h  n  y  �  |  m  ^  N  =  ,  !  #  %  &  #       �  �  �  e  <    �  �  \    �  u    �  e  '      �  �  �  (  9  1    �  �  U    �  �  q  5  �  �  u  2   �   �   ]  �  �  �  �  �  �  �  |  W    �  �  0  �  �  $  �  H  �  �  R  D  7  )      �  �  �  �  �  c  F  +    �  �  �  v     �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  u  \  B