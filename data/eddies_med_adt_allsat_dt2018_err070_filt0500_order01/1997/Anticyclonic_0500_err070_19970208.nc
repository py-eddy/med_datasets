CDF       
      obs    <   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?�M����      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       N��   max       P��      �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ���   max       =�"�      �  �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>\(��   max       @E�\(�     	`   |   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�fffff    max       @vo\(�     	`  )�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @+         max       @N�           x  3<   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�R        max       @�@          �  3�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �C�   max       >aG�      �  4�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A���   max       B,�1      �  5�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�v   max       B,�D      �  6�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       ?U�\   max       C�Hk      �  7t   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       ?=�   max       C�Xa      �  8d   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          �      �  9T   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          C      �  :D   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          %      �  ;4   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       N��   max       O��      �  <$   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�Y��|��   max       ?�֡a��f      �  =   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ���   max       =�;d      �  >   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>��Q�   max       @E\(��     	`  >�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?׮z�H    max       @vo\(�     	`  HT   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @*         max       @N�           x  Q�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�R        max       @���          �  R,   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         C5   max         C5      �  S   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�쿱[W?   max       ?����E�     �  T                  W   |         '         	         (               $                        0                     S               ?      	               ?   #   ,   
      	      6         �      N��O��Oh��N�O�u&PR�rP��O[�O�7eO�{O��&O�	�N�O�OA�O���N�VOmZANye�N` �P�ON�%N�0WO7j�O�N7MN5�N9iO���O�rNk�iN���N��aO��OCP%��ND�O@b�O:�QN6-HP-j�NU��N���NZ�O��:O��oN��O�P.O�vO��pN��O]�!NQ�wO�MO���N��N,L�PۄN���N�����㼼j�D��%   :�o;D��;ě�<o<49X<T��<�o<���<��
<�1<�1<ě�<ě�<���<���<���<���<���<�/<�/<�`B<�`B<�`B<�<�<�<��=o=+=C�=C�=\)=\)=�w='�='�=,1=49X=8Q�=@�=D��=H�9=H�9=L��=L��=P�`=}�=�o=�o=�+=�7L=�C�=�\)=��-=��-=�"�W[eghptytga[WWWWWWWWZ[\agt������}tpgb[Z������������������������������������������������

������BBKKDOh���������hOEB�����)5FNJ623EEB����������������������zx|���������������5NQX\]TNB5)���������������������#0DINPPLF90#
�������������������������

#)+(# 
�������
#-/7?C@</#
�����������
�����5468?BHOPUTVOLB;7655	"/;10//+("	��������������������VURT[]hmqlh[VVVVVVVV&#$%0BNgt�����tgB3)&ttw��������������|yt5305BNVURNB555555555`dgt������������tlg`858<HU_anypnjaUH<<88���������������������������������������"#/4<5/#""""""""""��������������������������$($�����������������
#/1/#
��� 

#%'%#
	
#,/220/+#
��������������������D>>Gm|���������zaZXD������������������|���������������~}z|�����
��'#).6ABCDB;6.)''''''MQg������������t_TOM" $)5;:65)""""""""""omryz~����������}zoo�������������������������������������'5>BCA=2'�������		������������������dfeht������������phd�������
!"
�����"()/4689862/)������������������������������������ 	
#+-20(#
��������	��������������� ��������������

������������������	 ����������������������������


���������������������������������������������������zÇÓàåéàÞÓÇ�z�n�c�a�_�a�j�n�z�z�(�;�4�,�(��������޽̽ܽ�����(�����ĺ������������������������������������)�5�9�=�<�+���������������������Z�s�����׾پ���ھʾ��f�M�D�<�7�@�M�Z����I�n�~ŔŰųŭŔŇ�n�b�<��ĦĞ�����������'�1�3�8�3�-�'�������ܹӹӹ迸�Ŀѿݿ�����ۿȿ������������������G�T�`�m�v�l�c�`�T�G�;�.�(�����"�.�G�������������������������x�s�m�s�w������������������������r�f�Y�R�Q�Y�`�r��������ûлڻллû�������������������������D�EEE
EEEEEED�D�D�D�D�D�D�D�D�D��`�m�x�y�{�}�y�x�m�`�T�G�D�@�A�D�G�J�T�`����	��!���	���׾���������������������������������ܻӻлɻлܻ����/�;�H�T�a�j�o�s�m�T�H�/�(��	���	��"�/����������������žŹŸŹ���������������һ_�l�x���������x�l�_�\�[�_�_�_�_�_�_�_�_�������	�"�7�A�H�A�B�;�/��	�������������6�:�B�D�J�K�G�B�6�)�����
�
���)�6�f�s���������s�f�`�_�f�f�f�f�f�f�f�f�f�Z�f�i�m�k�s�w�s�f�Z�M�A�4�2�/�2�:�A�H�Z������
����� ����������������������������	������������������������������ʼ˼̼ͼʼƼ�����������������������DIDVDbDgDgDbDVDIDCDIDIDIDIDIDIDIDIDIDIDI�M�Z�f�s�y�|�z�����s�f�Z�M�A�5�)�+�4�M�����ʾξ׾���������׾ʾ���������������*�,�4�*���������������������(�(�$�������������������
����������޼ۼ�������g�s�x�����������������������~�s�d�^�\�g�������'�(�'�������������ĦĳĿ����ĿĭĭĵĳĦēĆą�o�H�E�O�[Ħ�#�/�1�0�/�#�����#�#�#�#�#�#�#�#�#�#�A�N�Z�g�o�s�Z�N�A�5�(�������(�4�A�:�F�S�_�l�t�q�l�_�S�F�:�-�#���!�"�0�:������������������������Z�g�����������{�M�A�(�$�&� �"�&�5�A�N�Z�[�g�t���t�g�[�Y�S�[�[�[�[�[�[�[�[�[�[ŇŔŠŭůŹ������żŹŭŤŠŔŏŇńŇŇ�ݿ������ݿѿϿѿ׿ݿݿݿݿݿݿݿݿݿݽ����Ľн���ݽý����������y�s�s������ƎƚƧƳ���������������������ƧƚƄƊƎ�.�;�G�T�`�b�h�b�`�T�G�;�.�+�'�-�.�.�.�.ù������
�	�����������ùàÓËÈäìù�4�@�Y�b�`�Z�Y�M�@�4�'�������'�/�4�b�n�{ńŏœŏņ��{�n�b�U�I�@�?�<�B�R�b�e�r�~�~�����������������~�r�j�e�Y�M�d�e���������
� �
���������¿¼²±±·¿���<�H�U�^�V�U�H�<�1�0�<�<�<�<�<�<�<�<�<�<���������ʾ̾ʾɾ�������������~�����������������������������z�x�w�~������������DbDoD{D�D�D�D�D�D{DvDoDlDbD`DbDbDbDbDbDbǡǢǫǭǣǡǔǐǒǔǝǠǡǡǡǡǡǡǡǡ���ʼּ�������ּʼ������������������_�l�x�����������������������x�u�d�_�V�_E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E� Q . p O . 2 q @ 4 % + % x R > a E f * 3 ? , 0 < @ P i + 0 = M q > \ C V 5 W 9 j 7 A c U U z  9 S ! a I L < J h z 8 m {  F  K  N  =     �  
  �  �  z  e    �  J  �    �  ;  �  o  �  �  �  �  X  N  �  F  $  F  �  
  �  ;  [  V  "  �  �  g    m    /  �  �    !  �  M      V  u  ^  �  �  �  :  ��C��D��<e`B;��
<�h=�Q�>�<��=��=P�`=t�=<j<�h=L��=P�`=�%='�=D��=+=t�=y�#=H�9=C�=o=L��=o=C�=@�=��-=8Q�=�P=C�=t�=@�=@�=��=��=�+=��=@�=�
==P�`=]/=P�`=��P=y�#=��=�`B=�1=��=�t�=��=�t�=���=�=���=��P>aG�=�E�>JB	6>B	�rB"%B"�ZBkiB?B߬Bn�B~dB"9B��B%��B"GB�B�B�B^�A���B�BqB��B2	BͰB
\�BXYB1�B"��B�B2�B�B_B6�B$�B~`B ��A��0B�B9FB�B��B
��B�A��cB�B,B΃B2�B��B�cB4BN�B��B�:B$�VBGbB�~B�\B��B,�1BB	:�B	�B"WZB"�.BM]B@�B��B{�B��B9PB�B%��B"?�BAHB�yB�^B?�A�vB�B�]BB#BA�B��B
�B>�B@9B"�EB49B9OBE�BB�B�B$��B��B �?A�|�B�B@�B�&B��B
�B
�B >XB�gB+��B�uB@B��B��B��B?�B�*B�B$�UB@�B��B�-B��B,�DB@�A�+�A�@�A0��@%~�A���AG,QA�>�?[�Av0�Ad��A�x�@��@���C�P�Ah\gAQ�j@�[NA��^A�-�@�{,A��OA֝	AC\�A>(A�5]AӜ@��sC�yA>��ARA�:eA2;TA�A���?U�\A��>A�x�A�>�@�ԝ@X�A���A���A��A}��A!��B�Ae
4A��J@��jA���@�A���A�[AJ��A�{C���B��@���@���C�HkA�n2Aȁ�A0��@#��A��VAG�A�n8?=�Av��Ad�uA�bW@��i@�FC�WxAh�2AQ>�@��A�i�A��@��A�AցuACkA=�[A�|$AӀ'@��C�xA>t�AR��A���A4xA 'A�eJ?IE�A�q`A�^A�~@���@[�A���A��'A�A~��A#pB�Ad�A�M@��A�Q?�=�A� A�olAI��A��gC�� B��@�Ɨ@�� C�Xa                  X   |         '         
         (         	      $                        0                     T               @      
               ?   #   ,         	      6         �               !         /   C                           #               %                                             /               +            '         %                              %                           %                                                                                                      !            !                                             N��N�ъOgvN�O��O��O��ODUzOo�O`ϱOZQ�O'�nN�N��OA�O�kN���ON"Nye�N` �O��9O>JN�0WO7j�NSFN7MN5�N9iO/l(N� �Nk�iN���N��aO��OCOIqND�O@b�O!EN6-HO�fNU��N���NZ�O�h�O��oN��OU�N��AO��pN��OC9�NQ�wO�MO���N�N,L�O�N���N���  �  ~    �  �  ~  h  W  G  f  �    K  �  2  f  �  �  �    }  �  �  �  &  �  �  ^     �  5  �  �  Q  k    �    p    y  �  �  �  �  .  {  �  �  �  �    ,  A  
C    �  �  (  	��㼓t�:�o%   <#�
=,1=]/<#�
<�C�<�j<��
<���<��
<�j<�1<�<���<�/<���<���=�w<�/<�/<�/=t�<�`B<�`B<�=49X<��<��=o=+=C�=C�=��=\)=�w=0 �='�=m�h=49X=8Q�=@�=aG�=H�9=H�9=�hs=�%=P�`=}�=�+=�o=�+=�7L=�O�=�\)=�;d=��-=�"�W[eghptytga[WWWWWWWWebdgotv������tjgeeee�������������������������������������������������� ��������XVVZaht���������th`X������)-01.)�������������������������������������������)5DNOUUNLB:5)��������������������#0<>HIF@20##������������������������
#(*'#
�������
#-/7?C@</#
��������
��������56:BBBMORQTOJB=86655"/540..)$" 	��������������������VURT[]hmqlh[VVVVVVVV0.06=BN[tz}xtg[NB90}zu����������������}5305BNVURNB555555555`dgt������������tlg`<8<HU^\UH<<<<<<<<<<<���������������������������������������"#/4<5/#""""""""""�������������������������#'$������������������
#/1/#
��� 

#%'%#
	
#,/220/+#
��������������������STYamvz�������zma[WS������������������|���������������~}z|����������'#).6ABCDB;6.)''''''[X\gt�����������tga[" $)5;:65)""""""""""omryz~����������}zoo������������������������������� ��������'5>BCA=2'�������		�����������������nmot��������~tnnnnnn�������
!"
�����"()/4689862/)������������������������������������� 	
#+-20(#
��������	������������������������������

������������������ ����������������������������


���������������������������������������������������zÇÓÙàâàÓÓÇ�z�n�i�c�n�v�z�z�z�z���%� ���������ݽؽݽ������������ĺ�������������������������������������)�*�5�3�)�������������������f�s��������ƾǾ����������s�f�\�U�S�Z�f������#�0�I�[�c�^�5�������������������������'�-�3�6�3�)�'������ܹչչ迒�������Ŀѿܿۿ׿пĿ������������������.�G�T�`�f�h�f�`�V�T�G�;�1�.�%�"��!�*�.��������������������������������x�������r�������������������r�f�Y�X�Y�Y�f�q�r���ûлڻллû�������������������������D�D�EE	EEEEEED�D�D�D�D�D�D�D�D�D�`�m�x�y�{�}�y�x�m�`�T�G�D�@�A�D�G�J�T�`�������׾ʾ��������������������ʾ���������������ܻջлʻлܻ������;�H�T�a�h�m�p�m�a�T�H�/��	��	��"�1�;����������������žŹŸŹ���������������һ_�l�x���������x�l�_�\�[�_�_�_�_�_�_�_�_�����	��"�(�2�4�3�/�"��	� ��������������)�6�B�B�I�J�G�B�6�)����������f�s���������s�f�`�_�f�f�f�f�f�f�f�f�f�Z�f�i�m�k�s�w�s�f�Z�M�A�4�2�/�2�:�A�H�Z��������������������������������������������	������������������������������ʼ˼̼ͼʼƼ�����������������������DIDVDbDgDgDbDVDIDCDIDIDIDIDIDIDIDIDIDIDI�M�Z�f�n�s�u�t�u�u�s�f�Z�U�M�A�>�3�5�A�M���ʾ˾׾���������׾ʾ¾���������������*�,�4�*���������������������(�(�$�������������������
����������޼ۼ�������g�s�x�����������������������~�s�d�^�\�g�������'�(�'�������������ĚĩĳķķĳİĦĞĚčā�x�t�m�k�uāčĚ�#�/�1�0�/�#�����#�#�#�#�#�#�#�#�#�#�A�N�Z�g�o�s�Z�N�A�5�(�������(�4�A�:�F�S�_�k�l�s�o�l�_�S�F�:�-�"�%�-�2�:�:������������������������N�Z�g�y�����������g�N�A�<�5�,�-�2�;�A�N�[�g�t���t�g�[�Y�S�[�[�[�[�[�[�[�[�[�[ŇŔŠŭůŹ������żŹŭŤŠŔŏŇńŇŇ�ݿ������ݿѿϿѿ׿ݿݿݿݿݿݿݿݿݿݽ������̽ٽѽĽ������������x�x����������ƎƚƧƳ���������������������ƧƚƄƊƎ�.�;�G�T�`�b�h�b�`�T�G�;�.�+�'�-�.�.�.�.������������������������������ûðíù�ż4�@�M�S�R�M�C�@�4�+�'�%�'�.�4�4�4�4�4�4�b�n�{ńŏœŏņ��{�n�b�U�I�@�?�<�B�R�b�e�r�~�~�����������������~�r�j�e�Y�M�d�e������������ ������������¿³²¹¿�����<�H�U�^�V�U�H�<�1�0�<�<�<�<�<�<�<�<�<�<���������ʾ̾ʾɾ�������������~�����������������������������z�x�w�~������������DbDoD{D�D�D�D�D�D{DwDoDmDbDbDbDbDbDbDbDbǡǢǫǭǣǡǔǐǒǔǝǠǡǡǡǡǡǡǡǡ���ʼּ����������ּʼ����������������_�l�x�����������������������x�u�d�_�V�_E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E� Q D b O  6 R E %  &  x K > S J f * 3 9 $ 0 < E P i + ) 4 M q > \ C + 5 W 1 j  A c U N z  & ( ! a 7 L < J _ z ' m {  F  �  �  =  3  `  s  �  �  �  �  c  �    �    �  *  �  o  K  �  �  �  p  N  �  F  x    �  
  �  ;  [  �  "  �  V  g  �  m    /  �  �    �  �  M    �  V  u  ^  �  �  �  :  �  C5  C5  C5  C5  C5  C5  C5  C5  C5  C5  C5  C5  C5  C5  C5  C5  C5  C5  C5  C5  C5  C5  C5  C5  C5  C5  C5  C5  C5  C5  C5  C5  C5  C5  C5  C5  C5  C5  C5  C5  C5  C5  C5  C5  C5  C5  C5  C5  C5  C5  C5  C5  C5  C5  C5  C5  C5  C5  C5  C5  �  �  }  w  q  m  k  i  h  f  j  r  {  �  �  �  �  �  �  �  I  \  k  u  {  }  x  o  a  J  )  �  �  s  #  �  W  �  v   �  �  �  �  �      �  �  �  �  �  d  M  (  �  �  �  �  C  �  �  �  �  �  �  �  �  �  �  �  �  �  �  m  Q  5  '        #  m  �  �  �  �  �  �  �  �  ~  V  %  �  �  v  -  �  q    X  �  N  �  �  4  a  w  }  z  n  V  *  �  �  0  �  �  �  �  	�  
a  
�  
�  
�    Q  f  d  :  �  �    
~  	�  �  U  �  �  �  P  V  U  K  ;  +      �  �  �  h  /  �  �  q  N  5    �  �    +  <  E  F  B  ;  1  &      �  �  �  P  �  �  )  �  "  :  K  X  a  f  a  S  A  '    �  y    �  H  �  T  �  �  �  �  �  �  �  �  �  �  �  �  k  ?    �  �  g  &  �  �  �  �  �  �  �  �    �  �  �  �  �  y  R  '  �  �  7  �  +   �  K  7  "    �  �  �  �  �  f  D  $    �  �  �  j  /  �  �  �  �  �  }  [  0     �  �  �  ]  )  �  �  `    �  )  �  /  2  !    �  �  �  �  }  U  (  �  �  n    �  S  �  (  N  m    O  _  e  U  ;    �  �  X    �  k  A    �  K  w  �  )  �  �  �  �  �  �  �  f  7  �  �  w  0  �  �  ^    �  �  �  �  �  �  �  �  �  �  �  x  X  1    �  �  `    �  >  �  9  �  z  n  c  Y  O  B  5  &      �  �  �  �  j  A    �  �    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �    W  -    �    *  I  c  t  {  }  {  r  W  !  �  �  o  4  �  �  
  j  �  �  �  �  �  �  ~  q  \  D  )    �  �  �  _    �  �  E  �  �  �  �  �  �  �  �  �  �  }  g  O  7    �  �  �  =   �  �  �  �  �  �  �  �  �  �  �  �  ~  l  W  C  /  )  '  $  "  �  �  �  �  �  �    '  /  &    �  �  �  [    �  E  �  w  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �    {  x  u  �  �  �  �  �  �  �  �  �  �  ~  W  1    �  �  p  9     �  ^  S  J  9    �  �  �  �  s  A    �  �  N    �  )  �    �  �  �  �  �  �  �  �  �  �  W    �  d    �    �    k  �  �  �  �  �  �  �  �  �  �  k  N  )  �  �  �  =  �  {   �  5  *        �  �  �  �  �  �  s  V  6    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  }  v  p  i  b  [  T  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  ~  w  o  g  `  Q  9    �  �  �  �  r  `  K    �  �  }  N    �  �  M    k  a  X  Q  J  C  <  1  "    �  �  �  �  h  -  �  �  A   �    P  �  	"  	�  
D  
�  
�      
�  
�  
k  
  	�  �    �  �  �  �  �  �  �  �  �  �  �  �  �  z  q  h  V  5    �  �  �  �            �  �  �  �  �  r  7  �  �  K  �  �    �  �  G  h  p  l  d  U  A  '  	  �  �  t  2  �  �  �  o  X  M  N        �  �  �  �  �  �  �  �  �  r  d  [  Q  H  @  7  /  �  .  T  p  w  y  o  \  B     �  �  �  6  �  Z  �  �  �  F  �  �  �  �  �  �  �  �  �  x  o  f  ]  S  M  I  F  Y  u  �  �  �  �  �  �  {  X  5    �  �  �  �  �  y  `  ?  %  =  U  �  �  �  �  �  �  �  ~  o  `  Q  B  2  "       �  �  �  �  �  �  �  �  �  �  �  �  �  {  L    �  �  |  A      �  p  .  (          �  �  �  �  �  t  j  W  ?  "  �  �  o  =    {  k  Z  G  2       �  �  �  u  L    �  �  Q  �  �  1   �  �    g  �  �  �  �  �  �  �  �  V    �  W  �  K  k  ^  J    �    D  i  �  �  �  �  �  l  @    �  g    �  P  �  �  �  �  �  �  {  d  M  0    �  �  b  $  �  �    �  �  1  '  �  �  �  �  t  S  3    �  �  �  �  �  �  o  ,  �  }  <   �  E      �  �  �  �  �  p  G    �  �  i    �    �    t  ,      �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  A  5    �  �  �  �  �  �  f  0  �  �  �  S    �  �  �  �  
C  
  
  	�  	�  	�  	�  	�  	t  	F  	  �  r    �  7  �  c    �  ]  x  q  Z  >  #    �  �  �  f  8  4  1  �  �  j  %  �  �  �  �  �  �  �  �  �  �  y  m  [  B  *    �  �  �  �  m  J  Y    S  �  �  �  u  H    �  .  �  �    /    
�  	A  !  P  (  �  �  �  �  �  �  �  �  �  z  e  8    �  �  m  9     �  	  �  k  8  �  �  1  �  �  9  �  }  *    �  �  0  �  �  W