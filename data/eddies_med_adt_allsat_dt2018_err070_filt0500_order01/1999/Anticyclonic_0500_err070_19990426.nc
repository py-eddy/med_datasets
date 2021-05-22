CDF       
      obs    6   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?ļj~��#      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M�Q
   max       P�aA      �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       �49X   max       >bN      �  \   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>ٙ����   max       @Fb�\(��     p   4   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�         max       @v{33334     p  (�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @*         max       @Q�           l  1   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�?        max       @��           �  1�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ��o   max       >Y�      �  2X   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A��H   max       B1��      �  30   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�L�   max       B1�      �  4   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       ?� �   max       C���      �  4�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       ?��   max       C���      �  5�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          �      �  6�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          U      �  7h   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          ;      �  8@   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M�Q
   max       P�4�      �  9   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��!�R�=   max       ?��5�Xy>      �  9�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       �49X   max       >bN      �  :�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�(�\   max       @FO\(�     p  ;�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�         max       @v{33334     p  D   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @+         max       @Q            l  L�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�?        max       @�Y�          �  L�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         F\   max         F\      �  M�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�Fs����   max       ?뢜w�kQ     �  N�            #               @   	         (         (   
         d      l      B            
            F   	   �   A      K   )         3   2   �         7   	      	      
   	   /   OD9N�8�O�}P/�N���N�N�i�Nt�!P �IN�]�O?��Na�O���O:�YN8�|P�5O0�N���N���P�aAO-��Pn�N�'}P'TWM�Q
N��O��N��N��kOC��N���P�ʨO�;P'pO�N�Om��P��qP�fN͖�O1�P��O��O��IOq��O �~O���N�vjN�\�N�&lN��N7үN��OU�CN���49X�#�
�ě����
�D���o%   ;o;o;ě�<o<o<#�
<#�
<#�
<49X<D��<u<�o<�C�<��
<��
<��
<�1<�9X<�j<���<���<���<���<���<�`B<�<�=o=\)=#�
='�=49X=49X=H�9=H�9=P�`=Y�=ix�=�o=�o=�+=�+=�\)=��P=���=��T>bNidnouz���������zniiMCOS[hrt����tqh][OMM��������������������_ht������������tmh[_��
!#./0/%#
������������������������"$(+'"������������������������/@KH</
�����������������������	#/7<AFFA</#
	���������������������zyz|���������������SRUaxz}~���zna]\WUS��������������������#>HUnqcTUH>#=<>7CI\hru}xuh^\OGC=mnz{���������}~}znmmjhhjmouyzz}������zmjehn������BO7����zme�	')-03.)>::AN[gt������tg[NE>366BO[b^[OJBBB863333� 5BNgpvtmcB5)���������������������		)05765)						����)-6=4)����JOP[^ht���vth[OJJJJ��������������������siiit�������������ts #0013400#�����3<GNND=5���?<@DHPTacmsz�zzmaVH?[^\dm����������{h[O[�����1 ��������XSVX[ht���������th[X�����/KRPRNB5�������)BN[\NB5��������������������������������������������������065)�����������������������������������

������������������������~����������������~~����������  �������366BO[\da[YOBB663333����������������������
#&/7/*#
	�����xtuz�����zxxxxxxxxxxZ[ahjrt~vtnh[ZZZZZZunuz~��������zuuuuuu,%/9<HUnrqldaUH<60/,���������������������(�5�>�A�N�X�Z�\�\�Z�N�A�5�(�%�!�!�&�(�(�ʾ׾߾������׾ʾľ¾��������ɾʾ�����������������żŭťŭŰż������Z�l�x�����s�Z�F�A�<�(������(�4�L�Z�m�r�v�s�m�l�a�Y�T�R�H�@�F�H�M�T�a�f�m�m��� �$�(�����������������"�/�;�H�K�H�B�;�/�"����������;�<�C�C�;�.�"���"�.�8�;�;�;�;�;�;�;�;���������������y�`�J�?�E�U�`�r�h�y�������������������������������x�r�n�x�{�������5�A�N�S�Z�X�N�H�A�5�(������ �(�4�5��"�/�1�8�;�F�;�4�/�"����������������������	����	���������������������)�)��������������������������ʼּ�߼ּ̼ʼ��������¼ʼʼʼʼʼʼʼʾ����׾��
���۾ܾʾ����t�z�s�x�������	��"�.�8�.�%����	������������	�����������������������������������������(�5�A�N�W�Z�g�s�y�s�g�N�A�5�,���� �(�������H�m���������y�`�H�"��������������Ź������������������������ŹŭšūŭŴŹ���)�B�\�i�m�j�[�B�6�%��������������ʼּ׼�ݼ�ּʼɼƼ������������ʼʼʼʿ;�G�T�`�y�������y�T�G�;�.����� ��"�;�T�`�m�y�n�m�`�`�T�Q�T�T�T�T�T�T�T�T�T�T�Z�f�j�k�j�f�_�Z�M�J�A�G�M�Q�Z�Z�Z�Z�Z�Z����������������������������������������
���!�#�)�$�)�#���
�� ���
�
�
�
àêìù��������ùùìàØÖÖÛàààà�ݿ�����
�������ݿտѿοпѿ׿ݿݼ��������������ּּѼּ޼�������A�m�{�|�o�Z�N�5����ѿ������̿���ĚĦĳĿ��ĿľĵĳīĦĝĚččĊĊčĒĚ����4�Y�r�����{�i�M�'��ٻлŻлл���ÇÓÚÐÊÌÐÇ�z�m�a�`�[�I�I�R�a�h�nÇ�-�:�S�_�f�f�\�T�N�F�:�-�(�!�����(�-ƚ������)�8�3�&���ƧƁ�O�@�;�G�t�}ƈƚ�U�a�f�w�v�p�o�v�v�p�a�8�.�*�,�)�$�/�H�Uàìù��������������������ùõìàÛàà��������������������{�r�p�l�o�r��������0�<�E�I�L�O�b�i�0��� ����
����#�0āčĚĦĿ����������ĿĦĚčā�w�q�r�tāDoD{D�D�D�D�D�D�D�D�D�D�D�D{DoD`D[D_DaDo���������ннĽ����������}�y�r�y���������������������������~�y�r�m�r�r�~���������M�Z�c�n������������s�f�Z�M�A�4�/�1�@�M�ݿ�������ݿѿƿĿ��ĿĿѿҿݿݿݿ��6�7�B�D�O�Q�P�O�B�6�)����� �)�0�6�6��������������������������f�r�����z�r�f�b�^�f�f�f�f�f�f�f�f�f�f�3�4�@�J�L�Y�[�Y�L�@�<�3�-�/�3�3�3�3�3�3�-�:�F�O�S�W�S�I�F�:�3�-�,�+�-�-�-�-�-�-FF$F'F1F1F2F,FFFE�E�E�E�E�E�E�FFFEuE�E�E�E�E�E�E�E�E�EuEqEjEiEuEuEuEuEuEu 3 L < ! H e S < ; 4 " w K c Q 7 A p � _ + $ P @ ^ @ H b ) 6 3 7 X L � 6 e j s = t 9  7 + A & 0 � Q } / 2 M  @    �  �    A  �  �  �    �  �  �  �  c  �  �  �  �  	�  o  �  �  �    �  ,    �  �  �  �  s  D  �  �  F  3  A  F  <  �    �  !  k  �    �  )  �  �  �  练o�o<�j<�;�o%   ;�`B<o=�+<u<�<D��=H�9<���<e`B=L��<�9X<�9X<�`B==��>J<�/=�-<���<�=]/=C�=��=��=+=Ƨ�=��>#�
=Ƨ�=e`B=�x�=��T=y�#=��=���=ȴ9>Y�=��=��=��=��=�{=���=���=���=�9X>J>�wB�iB��B�B <�B4B31A��HB�BWB#/�B|B��B-�B��B".�B3B1��B��A���B��B)WB	2�Br�BV�B Q�Bc�B��B��B!�B
��B%S�B�A�'B�B�DBU�B��BۆB�B؜Be�B�B��B,nB��B+XB��BCJBB�Bp�B�ZB�,B~hB��BʝB�BB�IB A_BCPBcA�L�B6tB@0B#?tB9�B��BQZB��B"GB2�B1�B8�A��B@ B@B	@�BD�B@
B ?JBKYBSBK�B!�B:(B%>�B̚A�@B��BA�B)�B��B��B8�B�MB�NB�6B�B+BLuBE�B��B�B�B?B?�B��BCB0�A��AR�A��A;R�A�vjA�/�A� WAa�An�_@�NTA�I*A�5A�.A�I@���APAA[S�A��EA�Y�A���A��WA֕�@�;�Ad*nAi�5A>�$A��A�Z�A�[tA��A�&A��4A�3@�I�AǕW@~΅B�yA��sA�x@@��A��A�q�C�ϳA! �@`kA@��A|��A�T�A���@�:�?� �@~��C���C��A�yAR�mA���A;�A�rkAՋ�A��@Aa'�An��@��|A�`A���A��A��N@��AQ
�AZ��A���A�f�A��A�l�Aփ�@���Ac�Ai�A?TA���A�{�A�{UA�A��A�e�A��@��Aǀ�@�/B�A���AΆ[@���A��5A�h#C��,A!@�A@k{A{�1A�X�A�|c@�u-?��@��C���C��      	      $               A   
         (         (            e      l      C            
            F   	   �   B      K   )         3   2   �         7   
      
      
   
   /               +               1                     /            U      '      +                        =      /   %      ;   '         /   !                                                                                                9      !                              %               ;   %         +                                       OD9N�8�O��IN��N� LN�N�PENt�!O��N�]�N���Na�O���O�N8�|O�ǞN�ExN4�CN�U>P�V�O-��O��N|O���M�Q
N�SO\��N��N��kOC��N���O�gO�;O~�N�RO`B:P�4�P
;tN��kNӖ{P��O���O?�Oq��O �~O���N�vjN�\�N�&lN��N7үN��ODb5N��  �  �  Z  2  7  �    �  �  6  �  �  �  �  "    �  6  �  �  \  �  �      �  y  �  k  �  �  [  Y  m  �  �    Y  �  n  �  �  i  �  �  
]  �  �  g  �  �  �  
.  7�49X�#�
��o<��
�o�o:�o;o<�h;ě�<u<o<u<T��<#�
<���<�o<�o<�C�=D��<��
=D��<�j=D��<�9X<ě�<�h<���<���<���<���=aG�<�=� �=�t�=t�='�=0 �=8Q�=D��=P�`=aG�=��m=Y�=ix�=�o=�o=�+=�+=�\)=��P=���=���>bNidnouz���������zniiMCOS[hrt����tqh][OMM������������������������������������������
#,.#
���������������������������"#'*%"�����������������������
#/7<?<7/#�������������������#'/1;<@?<9/#��������������������|{|~���������������|UUanszz||~{znea_^ZUU��������������������"#2=CGHGB></#MGFMOQ\hinkh^\YOMMMM��������������������kiikmqvz|~������zrmk��������/6*�������	')-03.)AIO[gt������{tg[NGBA56;BO[[[OKFB>6555555 )5BNQVY[ZQB5)��������������������)-5653)����)256-)����JOP[^ht���vth[OJJJJ��������������������siiit�������������ts #0013400#������(-)�����?<@DHPTacmsz�zzmaVH?rlihkpt����������tr��������������������YTWY[ht���������th[Y����.KQPQNB5��������)BNZ[NB5�������������������������������������������������055)������������������������������������

������������������������~����������������~~����������  �������366BO[\da[YOBB663333����������������������
#&/7/*#
	�����xtuz�����zxxxxxxxxxxZ[ahjrt~vtnh[ZZZZZZunuz~��������zuuuuuu./<HU`anpokcaUH<71/.���������������������(�5�>�A�N�X�Z�\�\�Z�N�A�5�(�%�!�!�&�(�(�ʾ׾߾������׾ʾľ¾��������ɾʾ����������������������Ŷŷ����������A�M�O�Z�\�Z�T�M�A�4�2�*�4�5�A�A�A�A�A�A�m�o�t�q�m�j�a�T�I�I�O�T�a�k�m�m�m�m�m�m��� �$�(�����������������"�/�;�D�?�;�/�"������������;�<�C�C�;�.�"���"�.�8�;�;�;�;�;�;�;�;�������������������y�g�`�Z�W�Z�`�m�y�����������������������������x�r�n�x�{�������5�5�A�L�N�O�N�A�@�5�(������(�4�5�5��"�/�1�8�;�F�;�4�/�"��������������������
�����������������������������%���������������������������ʼּ�߼ּ̼ʼ��������¼ʼʼʼʼʼʼʼʾ����׾��������׾ʾ��������������������	��"�#�"����	����������������������������������������������������������(�5�A�N�U�Z�g�r�g�Z�N�A�5�)�����"�(�	�"�;�a�m�������g�T�H�/�"������������	Ź������������������������ŹŭšūŭŴŹ�)�B�O�\�b�^�O�B�6�)��������������)�ʼμּݼؼ׼ּּҼʼ������Ǽʼʼʼʼʼʿ"�.�;�G�T�]�g�l�n�m�`�T�G�;�.� ����"�T�`�m�y�n�m�`�`�T�Q�T�T�T�T�T�T�T�T�T�T�Z�f�h�j�h�f�Z�Z�X�M�D�I�M�T�Z�Z�Z�Z�Z�Z�����������������������������������������
���!�#�)�$�)�#���
�� ���
�
�
�
àêìù��������ùùìàØÖÖÛàààà�ݿ�����
�������ݿտѿοпѿ׿ݿݼ��������������ּּѼּ޼�������(�A�Z�g�k�g�Z�N�5��������������ĚĦĳĿ��ĿľĵĳīĦĝĚččĊĊčĒĚ���'�4�@�M�U�`�a�V�M�@�4������������n�z�~�{�z�n�n�k�a�^�a�k�n�n�n�n�n�n�n�n�-�:�S�_�d�e�[�S�M�F�B�:�-�+�����)�-Ƨ����#�0�1�$���ƧƁ�O�A�<�I�u�~ƉƖƧ�H�U�a�u�u�o�n�t�t�n�a�:�/�,�-�+�'�'�/�Hàìù��������������������ùöìàÞàà�������������������~�r�r�o�r�t�����0�B�I�K�N�b�g�0��
�����
��
��#�0čĚĦĳ����������ĿĳĦĚčā�z�t�tĂčD�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D{DtDsD{D����������ннĽ����������}�y�r�y���������������������������~�y�r�m�r�r�~���������M�Z�c�n������������s�f�Z�M�A�4�/�1�@�M�ݿ�������ݿѿƿĿ��ĿĿѿҿݿݿݿ��6�7�B�D�O�Q�P�O�B�6�)����� �)�0�6�6��������������������������f�r�����z�r�f�b�^�f�f�f�f�f�f�f�f�f�f�3�4�@�J�L�Y�[�Y�L�@�<�3�-�/�3�3�3�3�3�3�-�:�F�O�S�W�S�I�F�:�3�-�,�+�-�-�-�-�-�-FF$F1F2F+F$F"FFFE�E�E�E�E�E�E�FFFEuE�E�E�E�E�E�E�E�E�EuEqEjEiEuEuEuEuEuEu 3 L / " U e I < - 4  w L [ Q ; 6 2  C +  j 2 ^ B E b ) 6 3 # X > ` 6 f k o / u 2  7 + A & 0 � Q } / , M  @      �  �  A  �  �  :    �  �  V  B  c  "  �  =  n  z  o  �  v  )    �  �    �  �  �  -  s  
  I  �  '  �  .  �  �  y  �  �  !  k  �    �  )  �  �  �  �  F\  F\  F\  F\  F\  F\  F\  F\  F\  F\  F\  F\  F\  F\  F\  F\  F\  F\  F\  F\  F\  F\  F\  F\  F\  F\  F\  F\  F\  F\  F\  F\  F\  F\  F\  F\  F\  F\  F\  F\  F\  F\  F\  F\  F\  F\  F\  F\  F\  F\  F\  F\  F\  F\  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  r  d  W  I  �  �  �  �  �  �  �  �  �  �  �  �  ~  u  l  \  I  4      >  O  W  Z  X  P  B  *  	  �  �  W    �  �  {  Q  �  �    �  �  �  �      #  (  $        #  /  /    �  t  �  |  5  6  6  6  1  -  )  &  #        �  �  �  �  �  a  !   �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �                   �  �  �  �  �  �  �  �  �  }  ^  @  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  u  e  Q  =  )  �  �    ;  V  k  |  �  �  }  d  +  �  z    �  �  �  F  �  6  ,  "      �  �  �  �  �  �  �  �  �  r  ^  E  -  !    w  s  `  e  {  �  �    r  `  F  #  �  �  �  �  y  u  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  `  z  �  �  �  q  Y  >  "    �  �  �  �  B  �  �  0      �  �  �  �  �  �  �  �  �  �  n  D    �  �  �  ^    �  �  "  (  -  3  8  ;  6  1  ,  '        �  �  �  �  �  �  �  �  �  �  �  �  �        �  �  �  �  �  b  �  �  8  �    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  y  N     �   �  �  �    9  C  M  X  c  n  r  v  y  y  y  w  s  o  f  \  R  �  �  �  �  �  �  �  f  H  %    �  �  �  U  (    �  �  �  �  �  �  8  �  �  �  o    �  _    �  �  |  #  �  �  �  +  \  U  N  F  ;  ,      �  �  �  �  Y  !  �  �  �  r  0  k  u  �  D  p  �  �  }  L    �  �  ,  �  
�  	�  �  �       f  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  !  +  ,  /  =  Q  l  }  }  p  ]  H    �  t      �  &  4    	  
            �  �  �  �  �  �  �  �  n  S  8    �  �  �  �  �  �  �  �  ~  l  X  A    �  �  �  K     �   n  T  i  u  v  n  ^  F  &  �  �  �  <  �  �  K    �  �  D  �  �  �  �  �  �  x  W  3    �  �  p  >            �  �  k  f  ^  R  @  ,    �  �  �  �  �  `  I    �  i     �   c  �  �  �  �  �  �  y  d  O  C  6  )    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  m  X  @  )     �   �  �  �  �  �  �    H  Y  W  A    �  �  Z  �  j  �  $  J  V  Y  N  C  @  B  ?  0  !      �  �  �  �  �  �  o  U  >  &  	  
�  >  �  &  P  \  \  l  i  O  
  �  *  
�  
(  	D    x  �  �  	�  
�  Q    �  �    �  1  �  �    �  U  �  g    �  !�  �  �  �  �  �  �  �  s  Q  ,    �  �  �  w  =  �  �  �  �        �  �  �  `  G  !  �  �  �  �  _    �  "  b  _  $  E  Y  R  G  7    �  �  �  �  v  N    �  u    �  G  �  t  �  �  �  �  �  {  [  5    �  �  A  �  �  �  G    �  |  :  �  /  j  m  i  ^  N  5    �  �  �  3  �  J  �  �    �   �  v  �  m  ]  4    �  �  U  "  A  ?    �  w  �  _  �  �  N  �  �  �  �  �  �  l  /  �  �  =  �  i  �  f  �  0  �  �  �  �  �  ]  �  ~  �  8  a  h  b  D  	  �         �  �  	  	�  �  �  �  {  o  [  =    �  �  x    �  �  ]  &  �  \     �  �  d  F  )    �  �  �  ~  T  !  �  �  i  $  �  �    �   �  
]  
1  
  	�  	�  	�  	l  	8  �  �  n    �  8  �  N  �  E  0  �  �  �  �  �  �  �  �  z  m  _  P  @  /      �  �  �  s  6  �  �  �  d  �  b  ;    �  �  �  �  �    *  6  ?  C    �  g  [  P  E  :  !    �  �  �  �  �  r  [  >  !    �  �  z  �  �  �  �  ~  w  p  i  `  V  L  B  &  �  �  �  �  �  ~  p  �  �  �  �  p  I    �  �  �    �  u  (  �  �  5  �  �  6  �  �    q  a  L  6      �  �  �  �  s  V  :  %        
  
'  
  	�  	�  	n  	:  	  �  �  K  �  �  C  �  <  �  �    �  7  
  �  �  �  M    �  �  ]    �  �  U    �  |    �  �