CDF       
      obs    8   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?�\(��      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M���   max       P:�      �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ���   max       =\      �  l   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>��G�{   max       @E�\(�     �   L   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��
=p��    max       @vqG�z�     �  )   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @1�        max       @Q�           p  1�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @Ͱ        max       @�t�          �  2<   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �ě�   max       >         �  3   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A��W   max       B,�      �  3�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A��   max       B,�c      �  4�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >��P   max       C�u#      �  5�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >Ӟ�   max       C�w�      �  6�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          U      �  7|   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          /      �  8\   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          '      �  9<   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M���   max       P      �  :   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�N;�5�Y   max       ?׻/�V��      �  :�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ���   max       =\      �  ;�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�=p��
   max       @E���R     �  <�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ���Q�     max       @vqG�z�     �  E|   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @%         max       @Q�           p  N<   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @Ͱ        max       @�          �  N�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         ?   max         ?      �  O�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��_o�    max       ?׻/�V��     �  Pl         5                                                            .   	                     	      *                        "         U      
         '   	   
   ;            N�swN�F�O�QN��N�c�N`�N�@ N`5N`�N�`O�юP:�NG�sN�'�P7�N��
N�jN���N�?�OF��O'S�O��ZO��N�H0O�*N#�OY�O��O/`7O��N���N�k@OхdN8MuN���N�b�N��N�лNN'xOhнOC��N�N��6Ov4NvQ�N��Oi�@O�=_O��5N'�M���O�o}O&��N,x�N�ߌN@������ͼ�C��T���49X�49X�o;D��;��
;�`B<o<o<t�<t�<49X<49X<D��<T��<T��<e`B<e`B<e`B<u<u<�o<�o<�C�<�9X<�9X<�j<ě�<ě�<ě�<�/<�`B<�h<�h<�h<�h<�=o=+=\)=�w=#�
=,1=T��=]/=aG�=q��=u=�+=�C�=��-=��
=\ #03;<940)%#[QOS[bgptutg[[[[[[[[pjp���������������wpchhrt�����thcccccccce`cgt�����ztpgeeeeeefegt����tgfffffffffffa]]ehitt�����{tlhff��������������������)/5-) sstz�������tssssssss������&)(#����������6RY\UB)���������������������������������������������������
 "AB/����������������������������������������������������������������������������������������������������������������������������caefns�����������xhc"/;>>CEAB6&"!#%##,07<IIID<80#!)5BKNR[\_[UNGB5)����������������������������������������������������������������������
������!)5<BFDA5)�����������������������������������������������JKNR[gttthg[QNJJJJJJ~{����������������~~
#%//2:5//(#
"""#(/334/#$#AHJUanpzz}znaZUHAAAA����������������������������������������)&(/4<HR[ZSGINH<94/)��������������������=879@BOX[g^[SOCB====��������

�����7311;=GHONH;77777777��������������������������$)375)��)BOSYQHA<6)��������������:9<=HTQKH<::::::::::[bln{�}{nb[[[[[[[[[[z}�����������������z���������������������
#$##
���������������������������������������ҽ�����������ݽؽݽ������������������������������������������������������B�O�h�u�v�z�{Ă�u�^�[�Q�B�6�'����6�B�L�U�Y�e�i�k�e�Y�L�D�@�F�L�L�L�L�L�L�L�LÓàìøøðìàÓÑÇÂÇÏÓÓÓÓÓÓ��������������ýùú�������������������ż��������ʼ˼ʼ��������������������������n�{ŁŇ�{�x�n�b�_�b�b�l�n�n�n�n�n�n�n�n�Z�f�g�i�i�f�Z�M�I�M�O�W�Z�Z�Z�Z�Z�Z�Z�Z�h�u�xƁƍƁ�u�h�\�\�\�g�h�h�h�h�h�h�h�h�\�hƁƎƚƣƭƧƁ�u�h�\�O�E�<�9�:�O�S�\�������4�:�;�7�5�"��	���������������𿫿����ĿĿĿ̿ȿĿ����������������������/�;�H�T�W�V�T�J�H�;�/�/�#�(�/�/�/�/�/�/�/�;�T�]�b�v�z�w�o�a�H�;��	�����������/àìùü��ùùìàÓÍÒÓÔààààààE�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�����������������������������������������������������������������������������󹶹ùϹܹ�����������ܹϹù���������������������������������������������ҾM�Z�f�s�������������m�M�A�<�6�;�A�F�M�/�H�a�o�t�u�o�a�H�;�/��	�����������	�/�f�m�r�u�������������������r�i�f�^�a�f���������$�)�$�������������������������������	�
������������!�%�-�3�-�'�!������������4�A�K�M�O�M�D�A�4�.�(�������(�,�4���ʾ׾ھ����׾վ���������������������)�/�B�N�f�t�~�|�t�e�N�5�)�������`�m�r�y�z���y�u�m�a�`�Y�T�O�Q�T�T�\�`�`�����������������������������������������)�5�B�[�d�h�f�[�N�)���������������)�`�m�s�y���y�y�y�m�l�h�`�\�`�`�`�`�`�`�<�H�O�U�X�U�Q�H�?�<�/�%�#�!�#�&�/�6�<�<��(�*�,�3�5�A�A�A�5�(����
������Z�f�s�����������s�m�f�Z�W�Z�Z�Z�Z�Z�Z�A�G�N�S�Z�Q�N�A�A�5�,�,�2�5�9�@�A�A�A�A�����������������������������������������Ľн����	�
����߽Ľ�������������������������������������ùóôù�����޼����ʼռμʼ����������������������������лܻ�������������ܻۻлϻллл�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D����(�5�8�5�(�'�������������`�l�y���������������y�m�l�f�`�_�`�`�`�`��#�0�<�I�L�W�[�[�U�N�<�0�(�)�$�����~���������Ⱥ������������~�k�[�S�[�e�r�~���������������������������z�t�t�}������EEE(E*E2E*EEEEEEEEEEEEEE�����	�������������������������������������ʼ�������ּ����������r������'�4�>�<�0�'�������Իһܻ�������'ǡǭǵǰǭǣǡǔǓǔǖǟǡǡǡǡǡǡǡǡù������������������ùìèâìïùùùù�������������������z�������������������� N @ ( D 0 E L M P ^ ' @ c # 0 / ? - x 1 4 H X 2 r ~ 9 + I R B H Y t : [ � a 2 | L 2 < ' X ? > ^ 4 F � g ~ j . �  �  �  �  �  �  w  �  s  �  V  �  D  z  �     �  0  �  3  �  g  &    �  �  �  V  F  �  Y    �  L  �  �  �  �    b  �  �  1  �  �  }  �  �  �  �  G  ]  7  �  S  �  l�ě���C�=\);o���
�ě�<49X<t�<t�<D��<�/=\)<D��<T��=�P<�j<���<�C�<�t�<�/<�<�`B=y�#<�j<�1<�t�=C�<���=+=@�=o<�/=��<�=<j=+=o=C�<��=<j=�+=��=8Q�=��m=49X=P�`=�7L=��w=�v�=�C�=�O�>   =�j=��=���=��B%��B		�B9cB[�B	�uB	ژBUB[�B\qB
@B��B}�B�B�.B"B!��B��B<�BǉB�QB UB�HA��WB%�B��B�B d%B�|B�jB��B3NB�hB�	B��B��B�B˅B#�B)�B!vB�aB"��B�$B��A��B,�B�2Bb8BFbB!�B()B�|Bp�B)�BZGB,�B%�B	9B<�BD�B	��B	�B<EB>�B�qB
@B?�B5B%�B�'B�TB"9EBD�B?PB��B?+BHB��A��B%�AB�B�,B G}BUB�B�MBG!B�BA�B	-{B��B��B7�B]�B)� B �B�GB"�XB0�B�JA��B,�cB��B?B=|B>�B(8�B��B@�B=�B=�B,��A/��A��A�ٙ?��KA�h�A���@���A�2A?]�B�oB�1A���Aw��A���A���A�C�u#A�ŽA��H>��PA���AA��A��U@�tB��A��=@]�WA7��AO��A��Ai��At�hA��]Aj�|A�Y�A�y.AB�A�8�A �&A'��A�@���@�CC��]A���A�zA��@%�A��!C�x'@U��@���@�B�1A� v@�Y�A/�A���A�Z�?���A�O�A΄�@�0aA�X�A?�B�WBdsA��|Aw�A�|JA��A�NCC�w�A��8A���>Ӟ�A��AB��A�}�@�R�BtA���@T;*A91AP�-A��gAiB4At� A�n�AmiA�sA��ACjA��A!�]A'�TA�~@���@��@C��A��9A�A�w�@�lA�]aC�xl@S�@���@�޳B��A��@�_�      	   5                                                            /   	                     	      *                        #         U      
         '   
   
   <                     #                        #   -         /                        #                              %                                             !            )                                             #   '                                                               !                                             !            !            N�swN�F�ONaBN�N`�N�A�N`5N`�N�`O�юPNG�sN�E�O��0N��
N�jN2,�N[�dO[N��Ok�ZO�_�N�H0O�*N#�OY�O��O/`7O>'�N���N�k@O�B�N8MuN�]|N�b�N��N�лNN'xOhнN�	�N�N�>hN�6NvQ�NfhpOi�@O�=_O��5N'�M���O���N�@lN,x�N�n�N@�  ;    �  z  j  �  �  �  0  �  �  @  �  o  �  O    q  �  f  �  �  �  T    �  1  R  �  �  �  �  4    �  �  ?    m  �     �  6  |  D  �    �  D  "  �  	�  �  �     ������;�`B�D���#�
�49X��o;D��;��
;�`B<o<T��<t�<#�
<�j<49X<D��<e`B<e`B<�C�<��
<u<���<u<�o<�o<�C�<�9X<�9X<�`B<ě�<ě�<�`B<�/<�<�h<�h<�h<�h<�=@�=+=�P=��=#�
=49X=T��=]/=aG�=q��=u=��=��=��-=�1=\ #03;<940)%#[QOS[bgptutg[[[[[[[[����������������ehitt�����theeeeeeeefbdgt~���wtrgfffffffegt����tgffffffffffgb^`hqt����ztmhgggg��������������������)/5-) sstz�������tssssssss������&)(#������������6EQUOH6)��������������������������������������������������
�����������������������������������������������������������������������������������������������������������������������������dbfgou�����������thd"/45;?=::/"!#%##,07<IIID<80#!)5BKNR[\_[UNGB5)����������������������������������������������������������������������
������#)5@BCCB=5)%��������������������������������������������������JKNR[gttthg[QNJJJJJJ��������������������
#%//2:5//(#
"""#(/334/#$#AHJUanpzz}znaZUHAAAA����������������������������������������-./5<FHPMHA<6/------��������������������:9;BBOR[`[XONB::::::������

	���������7311;=GHONH;77777777��������������������������$)375)��)BOSYQHA<6)��������������:9<=HTQKH<::::::::::[bln{�}{nb[[[[[[[[[[�����������������������������������������
#$##
���������������������������������������ҽ�����������ݽؽݽ������������������������������������������������������B�O�Q�[�c�f�e�[�O�B�6�*�)�%�)�*�6�?�B�B�L�P�Y�e�g�j�e�Y�L�E�A�J�L�L�L�L�L�L�L�LÓàìôõìêàÖÓÇÅÇÑÓÓÓÓÓÓ��������������ýùú�������������������ż��������Ǽ������������������������������n�{ŁŇ�{�x�n�b�_�b�b�l�n�n�n�n�n�n�n�n�Z�f�g�i�i�f�Z�M�I�M�O�W�Z�Z�Z�Z�Z�Z�Z�Z�h�u�xƁƍƁ�u�h�\�\�\�g�h�h�h�h�h�h�h�h�\�hƁƎƚƣƭƧƁ�u�h�\�O�E�<�9�:�O�S�\�������	��"�/�6�,�"��	���������������俫�����ĿĿĿ̿ȿĿ����������������������/�;�H�Q�T�H�H�;�2�/�'�+�/�/�/�/�/�/�/�/�	��"�/�@�N�P�M�H�;�/�"��	��������� �	àìùü��ùùìàÓÍÒÓÔààààààE�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E��������������������������������������������������������������������������������ùϹܹ޹����߹ܹϹù������������������������������������������������������ҾM�Z�f�s�������������s�f�M�A�;�>�D�H�M�/�;�H�T�a�e�k�h�a�H�9�/�"��	����� �"�/�f�m�r�u�������������������r�i�f�^�a�f���������$�)�$�������������������������������	�
������������!�%�-�3�-�'�!������������4�A�K�M�O�M�D�A�4�.�(�������(�,�4���ʾ׾ھ����׾վ���������������������)�5�B�N�X�g�t�r�[�U�N�B�5�)������`�m�r�y�z���y�u�m�a�`�Y�T�O�Q�T�T�\�`�`�����������������������������������������)�5�B�Q�[�`�d�a�[�Q�5������������)�`�m�s�y���y�y�y�m�l�h�`�\�`�`�`�`�`�`�/�<�H�T�O�H�>�<�/�'�#�(�/�/�/�/�/�/�/�/��(�*�,�3�5�A�A�A�5�(����
������Z�f�s�����������s�m�f�Z�W�Z�Z�Z�Z�Z�Z�A�G�N�S�Z�Q�N�A�A�5�,�,�2�5�9�@�A�A�A�A�����������������������������������������Ľн����	�
����߽Ľ�������������������������	���������������������������뼽���ʼռμʼ����������������������������ܻ�������������޻ܻѻܻܻܻܻܻ�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D����(�5�8�5�(�'�������������l�y�����������y�p�l�k�k�l�l�l�l�l�l�l�l��#�0�<�I�L�W�[�[�U�N�<�0�(�)�$�����~���������Ⱥ������������~�k�[�S�[�e�r�~���������������������������z�t�t�}������EEE(E*E2E*EEEEEEEEEEEEEE�����	�������������������������������������ʼ������ּʼ����������������'�4�9�8�4�-�'������������"�'�'ǡǭǵǰǭǣǡǔǓǔǖǟǡǡǡǡǡǡǡǡù��������������ùìëåìöùùùùùù�������������������z�������������������� N @  @ 1 E K M P ^ ' 7 c $  / ? # p 5 % F [ 2 r ~ 9 + I K B H N t > [ � a 2 |  2 ;  X > > ^ 4 F � c \ j 2 �  �  �  A  �  ~  w  �  s  �  V  �  �  z  �  �  �  0  A  �  (  �  �  d  �  �  �  V  F  �  �    �  �  �  �  �  �    b  �    1  �  �  }  }  �  �  �  G  ]  �  �  S  �  l  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ;  '       �  �  �  �  �  �  n  Z  T  N  G  =  3  +  #        *  ,  )  %  !            �  �  �  �      �  �  �  �    A  f  �  �  �  �  �  �  �  �  c  0  �  s  �  �  b  W  s  �  �  �  �  �  �  �  u  h  U  >  !    �  �  }  6  �  f  g  i  j  j  j  i  f  a  \  O  ;  (    �  �  �  �  �  x  �  �  �  �  �  �  �  �  �  �  �  �  �  h  E  !  	   �   �   �  �  �  �  �  �  �  �  �  p  `  o  i  ^  U  M  Y  �  y  f  S  �  �  �  �    n  \  I  6  #    �  �  �  �  �  s  B     �  0  -  +  )  '  "      �  �  �  �  �  �  �  s  Z  A  (    �  �  �  �  �  r  ]  H  6  *        �  �  �  �  �  �  c  �  �  �  �  �  �  �  �  �  �  �  �  |  k  U  1     �  �  �    0  9  ?  ?  6  .  7  9  .    �  �  �  �    @  �  Q   �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  u  b  P  k  l  m  m  n  m  g  a  [  U  N  G  ?  8  0  "       �   �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  o  I    �  �  "  O  H  >  0      �  �  �  �  �  �  �  �  �  �  �    G  �      �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  n  o  p  p  q  q  p  o  m  l  i  e  a  ]  X  M  ?  1  #    �  �  �  �  �  �  �  �  �  �  �  �  �  u  j  _  H  .     �  T  ]  c  e  f  f  d  \  T  L  D  8  )    	  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  v  U  4    �  �  �  �  �  �  �  �  �  �  �  �  {  p  g  ^  R  /  �  �  @  �    X  �  �  �  �  {  Z  0  �  �  �  �  �    N  I     ~  T  S  R  Q  P  O  M  J  F  A  ;  3  +      �  �  �  3   �      �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  }  w  r  l  f  a  [  1  ,  '  "          �  �  �  l  E  $      �    �    R  L  E  >  7  0  )  #        
                 &  �  �  �  {  e  L  -    �  �  �  z  W  -     �  �  |  Y  5  �  �  �  �  �  �  �  �  �  �  �  f  >    �  �  1  �  5  7  �  �  �  �  �  ~  q  c  T  D  0    �  �  �  �  f  8     �  �  �  �  �  �  �  �  �  �  �  }  o  a  Q  ;  &     �   �   �    *  3  )    �  �  �  �  �  d  <    �  �  L  �  ]  �  K        �  �  �  �  �  �  �  �  �  �  m  R  6     �   �   �  �  �  �  �  �  �  �  �  �  �  �  m  H    �  �  �  ^  #  �  �  �  �  �  �  �  �  �  �  u  f  S  A  .              ?  ?  ?  @  @  @  A  >  :  6  1  -  )  %                       �  �  �  �  �  �  �  �  �  �  �  �    t  h  ]  m  g  a  \  V  P  J  D  ?  9  2  )  !            �   �   �  �  �  �  �  q  b  U  G  ;  E  I  =  -      �  �  �  �  �  �  �  
  �  �  �  �  �  �  �  �  �  b    �  ^    �  �  �  �  �  �  �  �  �  p  Y  B  +    �  �  �  �  �  �  �  p  \  #  ,  4  5  3  '    �  �  �  �  �  h  K  #  �  �  �  �  0  y  �  Y  �     3  ]  t  |  l  7  �  �    �  �    
5  �  �  D  >  7  1  *  "      �  �  �  �  �  �  �  �  l  T  =  %  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  j  1  �  �  �      �  �  �  �  �  �  �  �  u  i  Y  E  ,    �  �  �  �  �  �  p  M  (    �  �  �  �  �  �  v  L  #  �  �  �  h    D      �  �  ~  A       �  �  �  z  C  �  �  P  �  y  �  "    �  �  �  �  �  �  �  q  S  4    �  �  �  n  @    �  �  �  �  �  �  �  t  W  :    �  �  �  e  A    �  �  �  �  �  	Z  	�  	�  	n  	I  	  �  �  �  �  c    �  t  �  J  y  �  ;  �    ,  �  �  �  l  G      �  �  \    �  �  �  +  �    �  �  �  |  h  U  C  1  &  %  $  #      �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  _  /  �  �  f    �  1  �  G  �  �  �  �  i  G  %       �  �  �  �  �  �  �  �  W  *   �