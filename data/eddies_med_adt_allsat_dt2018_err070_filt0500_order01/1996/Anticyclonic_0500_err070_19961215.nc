CDF       
      obs    9   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?��/��w      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       Mܜ"   max       P���      �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ��`B   max       >hs      �  t   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�\(�   max       @E|(�\     �   X   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�z�G�     max       @vd          �  )@   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @2         max       @N�           t  2(   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @ɯ        max       @�J�          �  2�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ;o   max       >\(�      �  3�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�I   max       B*�)      �  4d   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A��   max       B*��      �  5H   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >�w�   max       C���      �  6,   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >B��   max       C��!      �  7   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          �      �  7�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          G      �  8�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          ?      �  9�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       Mܜ"   max       Pr �      �  :�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��s�Q   max       ?�i�B���      �  ;�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ��`B   max       >hs      �  <h   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�\(�   max       @E|(�\     �  =L   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�z�G�     max       @vd          �  F4   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @.         max       @N�           t  O   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @ɯ        max       @�!`          �  O�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         B�   max         B�      �  Pt   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?���u��"   max       ?�Ov_ح�     �  QX   
            	               )   t   )      ,   R                     J               q      
                                       �                  8            3   A      O      NR��N�ӰO�l�NW-�M��INV�NK��N�xAN��P!P��O�V�N�DMP'ԆP���N���NB�-NͱGMܜ"N��N~Pj��N	��N��NdF�N���PEOtN#�Np� OL}�N�fiO<2O�<AN*�2NV��O$�YO��N#6N���P
NQ�qP�*�N
4�N:a�N���N��'N,aP	�*N��
N�Nv��O�ΎO�j?M�*�O#JO��N7�,��`B�o;o<o<t�<D��<T��<�o<�o<�o<�t�<�t�<���<��
<�1<�9X<�9X<�j<�j<ě�<���<�/<�/<�h=o=o=+=C�=\)=t�=�P=��=��=��=�w=�w=�w=49X=8Q�=@�=T��=T��=]/=aG�=aG�=}�=�+=�+=�7L=�\)=� �=�9X=�;d=�x�>I�>O�>hsyv����������yyyyyyyy��������������������%5N[gitwwg[NB)'����������������������

�������������������������������)5595)��������������������ljnqz����zrnllllllll6Bh��������h[OB8�����5FCFMNB5����12/8EDO`bXYlythOB=:1�����	�����������
#-8=HPNKE</
������*5B[bc_age[N���
#$*+#

9200<AHIIH><99999999����������������������������������������#+000*'#��������������������#'5B[g����{}ytngUB#sqqt�����tssssssssss���������� �����BBOQ[hqsh[OBBBBBBBBB}uv�������������}}}}��������	������������������������������������������������	!"/;HJIC;/"	���
#%##
���������)-3)���BN[t�������i`c^[^[JB������������������������������������#/9<HSU_ajmjaUH=<4/#W[]fhkt�������toha[W��������������������36BO[f^[OB9633333333bcgs��������������gb��������������������xtv����!#������x;<HKMLJHE?=<;;;;;;;;97<HU_UUH<9999999999enqyz���������zneeee���������������������������������������������$$
�������jlljmz{}��|ztmjjjjjj))567;61-)!#/<=@@<<</*##!!!!!!�����)1872)�����������
"##
����#$'&#�������������������������

���������������������������L�Y�b�e�f�e�]�Y�M�L�E�C�L�L�L�L�L�L�L�L��������������������������ŹůŹ�������ҿ.�G�T�`�m�q�s�p�t�`�T�G�;�,�#�����.�'�3�:�@�B�A�@�3�,�'��!�'�'�'�'�'�'�'�'�����������������������������������������a�n�zÇÈÇÃ�z�q�n�k�a�]�Y�a�a�a�a�a�a�f�l�m�p�f�f�Z�U�T�X�Z�b�f�f�f�f�f�f�f�f�����������������w�s�o�s�z�����������#�����	������������������������������f�Z�O�L�L�W�Z�f����#�<�{Ňł�n�d�a�U�I�����Ķį�������
�#�'�4�Y�r����������Y�M�4�����޻����'�!�-�5�:�C�@�:�-�!��������!�!�!�!��������	���㾾��������������������������=�T�I����ƳƎ�h�O�8�9�^�\ƎƧ���m�y�����z�y�m�`�Y�Z�`�a�m�m�m�m�m�m�m�m�"�.�;�G�O�G�<�;�.�*�"��"�"�"�"�"�"�"�"������������������ݻ�����ûлٻллƻû������ûûûûûûûûûüY�f�r�����y�r�p�f�Y�T�Y�Y�Y�Y�Y�Y�Y�Y�M�Z�d�d�d�Z�M�H�A�:�A�C�M�M�M�M�M�M�M�M�O�hĉėďĆ�{�h������ìÕù����$�6�O�n�zÇÓÍÇ�z�n�l�m�n�n�n�n�n�n�n�n�n�n�	��"�.�;�G�I�T�T�T�G�G�;�8�.�"��	� �	�H�H�R�U�\�^�U�H�?�?�A�H�H�H�H�H�H�H�H�H�/�<�H�K�L�H�C�<�2�/�+�#�#�#�%�-�/�/�/�/ÔÚ��������������ùà�z�n�a�Z�Y�b�o�zÔ�������üü����������������������������������ùϹܹݹ�ܹϹù¹������������������T�a�f�k�s�z�{�~�����z�m�a�[�U�K�D�G�H�T��	��������������������� �����a�m�z���|�z�u�o�m�g�a�T�H�<�8�8�;�H�Q�a�U�U�c�e�X�;�/�	�������������"�0�;�?�H�U���'�/�(�'�����������������������������޿�������������)�,�=�5�4�)�'���������������-�3�:�F�K�S�_�a�_�S�F�:�8�*�#�!��!�&�-�#�/�2�<�>�=�<�/�*�#��#�#�#�#�#�#�#�#�#����������ݿܿٿٿݿ�����������(�7�A�H�R�T�N�B�8����������
���������������������������������������������ʼּ������̼����������o�\�S��������������������s�k�s��������������������������������������������������������������������������������������������������������ý��������������������������A�N�Z�^�`�Z�N�D�A�>�A�A�A�A�A�A�A�A�A�A������� ��"�#��	������������u��������ŠŭŹ��������ŹŭŠŔőŔŚŠŠŠŠŠŠ����������������������������������������ǡǤǭǣǡǔǈ�{�{�{�~ǈǔǖǡǡǡǡǡǡ²¼��¾´¦�t�h�f�j�i�`�g�{ŇŔŠŤťšŘŎŇ�{�n�b�U�K�O�[�d�n�{������������������������D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�EiEuE�E�E�E�E�E�E�E�E�E�EuEoEmEjEiEaEiEiE7ECEPEWEZEPECE<E7E3E7E7E7E7E7E7E7E7E7E7 A f * B T L k ) _ f N s 2 $ U B A 1 g _ . l [ d y V 6 4 { p > 3 ] 8 O < E : _ T ? I s P r j 7 Z j w I = = d ? / M    r  �  �  x  7  k  �  �  M  ,  �  �  �  �  �  �  b  �  *  H  �  �  5  (  �  �  o  "  �  �  �  �  ]  <  �  n  >  I  �  �  e  �  ^  o  !  �  I  �  �  x  �  �  S    r  ?  e;o;D��<��<e`B<�C�<�o<�o<���<�t�=e`B>1'=m�h<�=�%=��=��<�h=C�<���<���<��=���=+=�P=�w=@�>z�=�w=49X=m�h=#�
=e`B=��=0 �=0 �=]/=Y�=P�`=L��=��-=u>\(�=e`B=�7L=y�#=�+=��P=���=��P=���=ě�>V>2-=�F>\(�>-V>�wB-B��B�^B��B�Ba(Bo�B\DB+�B�	B��BνBzBuBo�B5KB�*B!��B"'�B%:"B:�B	�B	��B	B!:Bf�BOB"�"B��A�IB�B��B	�RB!fB��B��BOB��B��B�xB%�B��B4B?�B�B*�)BFBaA�@B>/BS�B��B-�B�mB��B�GBIBɣB��B�UB��B��BB�B�&B��B?�B��B�VB�B�B�B;�B=�B�B!]�B"0�B%>sB/�B	E�B	�QB?�BIWB�!B�B"�4B�mA��B��B\�B
G�B!@�B[aB��B?�B��B��B8RB?�B��BMZB:
B6�B*��B�dB?�A�~�B=eBA�B�B 8B�wB�B@BG�?�AFA��Ad��?���A��rAǭA@tQAE�A��BAF<A��@��0@m AP��B�^Ak=�Ab3�@�)�@�4%@ߎ�A=�^A�(AȿWA`��AģA��A��@���>�w�A���A�/�A�,A�Ծ?�s�A�HA��@~*eA�o�A~�A��)A�`�@��	A��A�EA�`�A!��A���A���A���@�}B/�A�Z}A�DA���C���C���C���?Ս�A���Ad'?�F�A��A��A?	�AE�A���AD�1A�}�@��g@k@�AOV�B@IAk�Ab�@�3m@��A@��RA=��A�w�A�t�A_	�A�k�A1Ả@�Ww>B��A��A�i�A��7A��<?�K!A�
�A�	@�A�|�A�A�{�A���@� A�2�A�}-A�:BA *�A��A���A��Q@��Bl�A���A�1A�OxC�؋C��!C���   
            	               )   t   )      ,   S                     J      	         q      
                                    	   �               	   8            4   B      O                !                     -   9   -      )   G                     ?               /                  +                     '      9                  -            #                                             -      #      #   9                     ?               !                  #                     '      !                  -            #               NR��N�ӰO�:3NW-�M��INV�NK��N�xAN��P!O�b�O�,YN�DMO���Pr �N���NB�-NͱGMܜ"N��N~Pj��N	��N��NdF�N�SO�UvN#�Np� OL}�N�fiO<2O�1N*�2NV��Nc�N��N#6N���P
NQ�qO��PN
4�N:a�N���N��'N,aP	�*N��
N�Nv��O�ΎO�j?M�*�O)yO��N7�,    �  ~  �  �  -  �  �  �  �  
7  v  �  �  �  �    �  w  a  D  �  t  m  ;  �  �  �  �  �  �    �  o  �  X  9  q    �  �  +  �  O    K  s    +  <  �  �  R  �  �  
#  ���`B�o<#�
<o<t�<D��<T��<�o<�o<�o=���<���<���<�=�w<�9X<�9X<�j<�j<ě�<���<�/<�/<�h=o=C�=�7L=C�=\)=t�=�P=��=,1=��=�w=<j=#�
=49X=8Q�=@�=T��=�F=]/=aG�=aG�=}�=�+=�+=�7L=�\)=� �=�9X=�;d=�x�>O�>O�>hsyv����������yyyyyyyy��������������������$ ")5NV[gkk_[NB5)$����������������������

�������������������������������)5595)��������������������ljnqz����zrnllllllll6Bh��������h[OB8�����)046750)��49BOSUhnpotohOG@>664�����	������������
#/8ADGFC<#
�����)6NW\YZ_[NB,	��
#$*+#

9200<AHIIH><99999999����������������������������������������#+000*'#��������������������#'5B[g����{}ytngUB#sqqt�����tssssssssss���������� �����BBOQ[hqsh[OBBBBBBBBB�y����������������������������	������������������������������������������������	!"/;HJIC;/"	���
#%##
���������)-3)���]gt���������sdfa`e[]������������������������������������G@HUadfaUHGGGGGGGGGGZ[^ghlt�������tkhe[Z��������������������36BO[f^[OB9633333333bcgs��������������gb��������������������������������������;<HKMLJHE?=<;;;;;;;;97<HU_UUH<9999999999enqyz���������zneeee���������������������������������������������$$
�������jlljmz{}��|ztmjjjjjj))567;61-)!#/<=@@<<</*##!!!!!!�����)1872)�����������
"##
����#$'&#�������������������������

���������������������������L�Y�b�e�f�e�]�Y�M�L�E�C�L�L�L�L�L�L�L�L��������������������������ŹůŹ�������ҿ.�;�G�T�`�g�k�i�`�W�M�G�;�.�+�$�� �'�.�'�3�:�@�B�A�@�3�,�'��!�'�'�'�'�'�'�'�'�����������������������������������������a�n�zÇÈÇÃ�z�q�n�k�a�]�Y�a�a�a�a�a�a�f�l�m�p�f�f�Z�U�T�X�Z�b�f�f�f�f�f�f�f�f�����������������w�s�o�s�z�����������#�����	������������������������������f�Z�O�L�L�W�Z�f�����#�0�<�I�P�O�I�>�0�#�
�������������
��Y�f�q�w�r�M�4�'������������'�4�Y�!�-�5�:�C�@�:�-�!��������!�!�!�!���ʾ���������׾���������������������� ��"�$����ƧƎ�h�Q�L�T�hƁƎƧ�̿m�y�����z�y�m�`�Y�Z�`�a�m�m�m�m�m�m�m�m�"�.�;�G�O�G�<�;�.�*�"��"�"�"�"�"�"�"�"������������������ݻ�����ûлٻллƻû������ûûûûûûûûûüY�f�r�����y�r�p�f�Y�T�Y�Y�Y�Y�Y�Y�Y�Y�M�Z�d�d�d�Z�M�H�A�:�A�C�M�M�M�M�M�M�M�M�O�hĉėďĆ�{�h������ìÕù����$�6�O�n�zÇÓÍÇ�z�n�l�m�n�n�n�n�n�n�n�n�n�n�	��"�.�;�G�I�T�T�T�G�G�;�8�.�"��	� �	�H�H�R�U�\�^�U�H�?�?�A�H�H�H�H�H�H�H�H�H�/�<�H�I�K�H�B�<�1�/�-�$�&�/�/�/�/�/�/�/ÇÓìù������������ùÓÇ�z�p�j�k�u�zÇ�������üü����������������������������������ùϹܹݹ�ܹϹù¹������������������T�a�f�k�s�z�{�~�����z�m�a�[�U�K�D�G�H�T��	��������������������� �����a�m�z���|�z�u�o�m�g�a�T�H�<�8�8�;�H�Q�a�T�]�^�R�H�;�/�	���������������"�/�;�H�T���'�/�(�'�����������������������������޿�������������'�'� ����������������-�/�:�F�J�S�^�`�_�S�N�F�:�+�$�!� �!�+�-�#�/�2�<�>�=�<�/�*�#��#�#�#�#�#�#�#�#�#����������ݿܿٿٿݿ�����������(�7�A�H�R�T�N�B�8����������
���������������������������������������������ʼּۼ��߼ڼѼ����������������������������������s�k�s��������������������������������������������������������������������������������������������������������ý��������������������������A�N�Z�^�`�Z�N�D�A�>�A�A�A�A�A�A�A�A�A�A������� ��"�#��	������������u��������ŠŭŹ��������ŹŭŠŔőŔŚŠŠŠŠŠŠ����������������������������������������ǡǤǭǣǡǔǈ�{�{�{�~ǈǔǖǡǡǡǡǡǡ²¼��¾´¦�t�h�f�j�i�`�g�{ŇŔŠŤťšŘŎŇ�{�n�b�U�K�O�[�d�n�{������������������������D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�EiEuE�E�E�E�E�E�E�E�E�E�EuEoEmEjEiEaEiEiE7ECEPEWEZEPECE<E7E3E7E7E7E7E7E7E7E7E7E7 A f % B T L k ) _ f ; l 2  V B A 1 g _ . l [ d y V 4 4 { p > 3 \ 8 O ; C : _ T ? 4 s P r j 7 Z j w I = = d 9 / M    r  �    x  7  k  �  �  M  ,  �  �  �    d  �  b  �  *  H  �  �  5  (  �  �    "  �  �  �  �  �  <  �  f  0  I  �  �  e  �  ^  o  !  �  I  �  �  x  �  �  S    Y  ?  e  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�      �  �  �  �  �  �  y  b  K  6  #      �  �  �  �  �  �  �  �  �  t  h  \  P  D  9  .  $     "  $  &  (  +  -  0  !  K  _  q  z  ~  }  w  m  a  R  >     �  �  v    �  4  �  �    }  {  q  f  [  O  B  6  '      �  �  �  �  u  A    �  �  �  �  �  �  x  a  I  0    �  �  �  �  �  q  T  5    -  .  .  .  .  1  :  C  M  V  ]  b  h  m  r  x  ~  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  w  c  N  9  $     �  �  |  y  u  r  n  j  e  ^  W  P  J  C  9  +       �   �   �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  o  [  I  7  -  -  %    �  �  c  �  g  �   �  �  ~  �  	O  	�  	�  	�  
  
0  
0  
7  
$  	�  	�  	'  �  �  �  D  :    0  n  v  p  _  D    �  �  b  �  |  C    �  t    �  �  �  �  �  �  �  z  k  Z  F  2      �  �  �  �  �  �  �  �  "  S  |  �  �  �  �  |  \  8    �  �  (  �  k    _  �  y  �  ^  }  �  �  �  a    �  x  Y  h  I    �  ?  �  �  �  j  �  �  �  �  �  w  [  <    �  �  �  U  "  �  �  �  Q  �  �          �  �  �  �  �  h  E     �  �  �  �  `  ;     �  �  �  �  �  �  �  �  z  p  e  Z  O  C  7  )    �  �    M  w  p  h  a  Z  S  K  A  4  (        �  �  �  �  �  �  �  a  X  N  E  <  3  *  !                      !  #  D  ;  2  *  !        �  �  �  �  �  �  �  �  �  �  �  �  �  C      8  '    Y  _  >    �  �  Q  �  h  �  b  q  f  t  d  T  C  7  ,           �  �  �  �  ]    �  �  ]  &  m  U  =  #    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  ;  .         �  �  �  �  �  �  �  �  �  �  �  �  �    {  �  �  �  �  �  �  �  �  �  s  Q  /    �  �  �  n  4  �  �  
�  �  "  q  �  �  �  �  �  �  O    �  K  
�  	�  �  V  �    �  �  �  �  �  �  �  �  �  �  x  h  O  .    �  �  �  �  �  �  �  �  �  �  �  �  f  =  .  3    �  �  y  ?    �  �  B  �  �  �  �  �  �  v  h  Y  F  >  ;  *  �  �  J  �  h  �  R  �  �  �  �  �  �  �  �  �  �  �  �  �  �  {  s  k  c  [  S      �  �  �  �  �  �  s  T  5    �  �  �  �  ^  0  �  @  �  �  �  �  �  |  g  `  R  E  L  H  >  (    �  �  C    �  o  ^  M  <  *    �  �  �  �  �  t  d  Z  Q  G  G  I  K  M  �  �  ~  s  i  ^  T  J  @  6  ,  "        �  �  �  �  �  q  �  �  �  �    (  =  M  W  W  Q  H  ;  *    �  �  �  �  -  6  1  (  &    	  �  �  �  �  �  �  b  0  �  }  +  �  �  q  o  n  h  Y  I  2    �  �  �  �  m  H  #  �  �  �  �  ]       �  �  �  �  �  �  �  �  �    n  \  J  9  &       �  �  �  �  �  �  �  {  e  M  1    �  �  �  p  B    �    A  �  �  �  �  }  s  g  \  M  =  .         �  �  �  �  �  �  �  �    �  L  �    #  *    �  I  �  �  �  
�  	�  I  o  �  �  ~  z  w  s  o  k  h  d  `  \  W  R  N  I  D  @  ;  6  1  O  P  R  U  [  [  J  8  "  �  �  {  >  �  �  [    �  Y  �        �  �  �  �  �  �  �  �  j  T  ;       �  �  �  �  K  G  D  @  <  8  2  -  '  !  "  (  /  6  =  +    �  �  �  s  c  S  A  0      �  �  �  �  �  a  7    �  �  Q  �  Q      �  �  �  �  �  L    �  m    �    �  H  �  ,  .  ?  +    	  �  �  �  �  �  �  �  �  �  �  �  u  a  M  =  -    <  6  /  )  %  .  6  >  @  9  2  +        �  �  �  �  �  �  �  �  l  R  6    �  �  �  �  �  u  Z  <    �  �  �  �  �  �  s  9  
  �  -  -  2    �  �  y  '  �  2  �  �  5  �  R    �  �  j  2  �  �  \  
  
�  
F  	�  	U  �  1  l  _  '  �  �  �  �  �  k  C    �  �  �  p  C    �  �  �  V  "  �  �  D  �  a  ?    �  �  �  :  �  0  �  �  �  u  %  	�    �  �  
#  	�  	�  	�  	�  	q  	<  �  �  ^  �  �    �     u  �  1  	  �  �  �  b  >    �  �  �  �  �  d  @  	  �  f    �  0  �  Y