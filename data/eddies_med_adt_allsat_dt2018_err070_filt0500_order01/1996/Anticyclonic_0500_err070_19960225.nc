CDF       
      obs    <   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?��l�C��      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M�5   max       P�g.      �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       �0 �   max       =���      �  �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�z�G�   max       @F"�\(��     	`   |   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��z�G�    max       @vo33334     	`  )�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @.         max       @O            x  3<   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @��          �  3�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �#�
   max       >aG�      �  4�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A���   max       B0h      �  5�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�iU   max       B/�A      �  6�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       =8   max       C�g      �  7t   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       =^:�   max       C�g�      �  8d   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          �      �  9T   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          E      �  :D   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          )      �  ;4   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M�5   max       P�      �  <$   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�4�J�   max       ?��u%F      �  =   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       �0 �   max       =��m      �  >   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�Q��   max       @F"�\(��     	`  >�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��z�G�    max       @vo33334     	`  HT   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @.         max       @O            x  Q�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @          �  R,   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         A�   max         A�      �  S   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�L�_��   max       ?���O�;e     �  T                  
      	            
      X   J      	   �         "   ,      
         !   
   	      0         "                  G                                                   #   W   c   N�N��eN�$O=�NA��N-4�N?�NN�N�=OhW N�kSN?9P7�aP�g.PE�NHR�M�Pr�N�1�M�5P
�PfW�NZN�O-��O9�O�&NqxN���O�zPIkO���Od��O��jNߍ�O�[N�j�N��N���Pe�O!O�N&�NQwfN^$�N�7O��O���N�}�N�W2O��SO���OT�NA��O1�N�48O��NǢEOf�3O�;�N��$�0 ż�`B��`B��t��u�49X�49X�#�
��`B�ě����
���
��o�D���o��o%   ;D��<o<t�<e`B<u<u<�t�<�t�<�t�<�1<�j<ě�<���<�`B<�`B<�`B<�h<�h=o=o=\)=t�=�w=#�
=#�
=0 �=0 �=0 �=@�=@�=L��=ix�=m�h=}�=��=�+=�C�=�\)=�t�=��P=\=���=�����������������������fcbggkt����~zvtgffff

#)'#








miot������������ttmm��������������������||�������||||||||||������������������������������������������������������������������������������������������������������������������������=<EO_h����������h[B=���N`dbhlbN�����������"#!��������������������������}������������������)1B[���������t`[NF+)"#0<@C=<0#"""""""")*-+)#/49HLMH</#������5Nijef_N5�����������������������*15BN[^_^[NB<5******
#',/0/-#
fegnqz����������~znf� 	"/9BHKD>;0/. �������


����������RX[gt���tg[RRRRRRRR�������	����������)KN[N5)������):DI@BNONB50&��
#/<HSMJHB6-#
UWYSH@</##*/<HMTU(')/<HMU`WUHD<3/((((��������������������	
 "####
#%/00/# *56CCKFC60*#����)5;==6+���������������������()696);99;HIRTUTIHG;;;;;;;��������������������mptt���������tmmmmmm�������������������������)7:6)����pt�����������tpppppp��������� ����������/?B>/������QTamtz}������zmiebTQ
)68635.)!��������������������#./3764/,#"./4;@?;9//"��������������������.%'./<HTSMH=</......����������	��������������������������)02,)
���������������������������������������������U�a�n�z�{ÇÓËÇ�z�v�n�a�U�O�N�U�U�U�U�x�~���������������x�x�x�x�x�x�x�x�x�x�x�[�g�t�w�z�v�t�s�n�g�[�Z�Q�N�J�G�N�O�[�[���������������������������������������ٹ����ù͹ȹù�����������������������������������
������������������������'�,�3�5�3�.�'�#����!�'�'�'�'�'�'�'�'���������ùùù������������������������������1�D�M�R�M�>�4�'������������������������������������������������Ѻ��������ɺ�������������������������������(�A�M�f�y�s�K�A�4�(���Խ߽ܽ����Ƨ����,�&�0�.�$�����Ǝ�h�_�>�I�uƚƧƧ�zì����������������ùìÖ�n�[�U�a�z�A�N�Z�a�\�Z�N�A�5�2�5�?�A�A�A�A�A�A�A�AE�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E��O�tĖěčā�q�[�O�6�2� � ������6�O�r�������������r�r�k�o�r�r�r�r�r�r�r�r�������������������������������������.�G�m�����Ŀ̿οʿ������y�m�T�7���"�.�Ŀѿۿ����6�3������ѿ�������������Óàèìîùý������ÿùìàÓÌÍÊÓÓ���
�����
��������������������������������4�5�6�5�&��������޿߿��������������������������������������T�a�h�s�t�m�a�H�;�"����������/�;�H�T�U�a�j�n�o�n�n�a�U�T�J�T�U�U�U�U�U�U�U�U�y�������������|�y�o�p�p�y�y�y�y�y�y�y�y�Ϲܹ��� � �������ܹعϹù����ùɹ��/�;�T�`�b�\�T�/������������������	��/�s�y�{������s�Z�M�H�7�(��� �4�A�F�Z�s�s�������������������������������q�g�i�s�5�)��������������)�5�B�K�W�S�N�B�5������	�������������������������������(�5�A�N�T�Z�g�k�g�N�A�5�(�"�������(�4�>�A�M�Z�d�Z�M�F�A�4�1�������/�;�;�=�;�0�/�"�!��"�-�/�/�/�/�/�/�/�/�����������������y�m�k�k�m�q�y������������������D�T�U�b�U�P�<�0�#�
����ĸĶĻ��ŇŔŚŠťŪũŬŠŔŇ�|�{�{�x�{�~ņŇŇ�'�,�3�=�3�3�*�'�#�� �%�'�'�'�'�'�'�'�'����%������ ������������������������������������������������������"�.�;�;�D�G�I�G�;�.�"����"�"�"�"�"�"���������������������������������������������"�/�3�5�0�(������������������������������������������������������������������ʾ׾����	��	����׾ʾ����������s�w�����������������s�l�e�[�\�f�m�q�s��������������ŹŭŠřŒŗŠŹ������e�r�~���������������������~�r�j�e�b�a�e�z�������������z�o�n�z�z�z�z�z�z�z�z�z�z����*�C�O�R�O�L�C�A�6�*���	�������#�0�7�<�=�<�;�0�#���
�
���
��#�#�#���������Ƚǽ˽ɽ½���������������������D�EEEEE(E*EEEE D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D|D{D�D�D��r����������������{�i�f�Y�M�D�D�G�R�g�r�Y�_�f�k�r�t�v�r�f�Y�W�O�M�K�M�S�Y�Y�Y�Y B f ` K > O E F L b f 8 / @ L d N @ 9 \ c E 7 M K , A D ` , A o q , 2 \ m I R S G k U 3 A d  | g j u ? Z J > u : F F f  /  �  Q  P  `  S  j  P  I  B  �  X  $  d  �  \    �  �    �  ;  �    �  �  �  w  �  P  z  �  6  �    m    3  �  �  n  �  g  s  �  �  �    l    �  e  Z  }    �  �  �  �  �#�
��t��ě��ě��#�
��o���
�o:�o<D��;o;��
<�/=�1=�hs;�o<o>aG�<e`B<D��=@�=m�h<�`B<�`B=�P=t�=]/=+=o=�P=���=8Q�=,1=}�=T��=L��=49X=�w=#�
=�G�=q��=<j=@�=<j=L��=u=���=u=�C�=�9X=�{=�v�=�t�=� �=�{=ȴ9=�/>:^5>J��=��B�B	��B$�
B
�B�tB��B �B!=B@B!�<BhRB"BB��B@�Be,B)�B�B	8B%��B�fB�
B�3B!�B��BYZBz�A���BI�B	^�B�B��Ba�B�B�yB�bB�gB$��B�ZB0hB�Be�B�oA�h�BYBB�kB �B�B;BQ�B��A���B��B(�B��A�$B+bCB�;BsB��Bn$BK�B	��B$��B
B/B��B��B �B!?PB@RB"8�B�&B"8+B�rB��BXOB$+B=�BčB&#3B��B5B��B"@&B9 BKB��A�sRB@"B	�B��B�KB��BA�B�.B�B�XB$��B��B/�AB@�B?�B��A�xBU�B��B�BP�B�YB�B �A��?B� B@�B�0A�iUB*�@B�4B��B�RB�:A�o&AǄ�@��A�,�B��=�vA?D�=?�Oe=8@���A��@$W�A6�7B��A��4A�@�C�gA�~^@�2A�|zAkحA�|A�"A���A��A�.A��YA�`�Am�h?A��A@�8A�V�A��A�CA�lA9�A�tAn�A�tA�<�?�?oA�P�A���Aa#�A�{A�u�@�wATf�ADK
A���@y�A�W�A�?�A�M�A!��C�eC��@��W@��A�~�AǄ�@�KA���B�%=��@?Qk�?���=^:�@�A���@#�0A6��B=�A�wQA�HWC�g�Aڅ�@��A�x�Ak��A��A�q�A�|5A��\A�q�A�|�A�XAm,m?o�A�AA?!A�}wA���A��A�aA9a?A�l�An��A�|�A�rB?�>%A���A��A`��A��DA�nw@�<�AS$AE�A�c�@�A�3A�m�A�|�A!�C�g�C���@��H@���                        	   	                X   J      	   �         "   ,               "      	      0         "                  H                                                   #   X   d                                          /   E   /         3         +   3               #            3   %                        '                     #                                    !                                          '      %                  )   '                                                                           #                                       N�N��eN�$N��MNA��N-4�N?�NN�N�=O�"N�kSN?9PW�O�)HPd�NHR�M�ORݼN�1�M�5O��>P�NZN�OyeOr�O�9�NqxN���N�\�O�G�N���Od��Oo&N�)O�[N�j�N��N���O���N�_6N&�NQwfN^$�N�7O��O���N�}�N�W2O���O\�O B�NA��O1�N�48O��NǢEOf�3O�T�N��$  P  �  �  0  :  �  �  �  -  t  q  �    X  	]  B  <  D  �  �  �  �  i    �    �  P  �  w  u  �  �  c  �  �  o  �  �  
]  M  �  �  K  ~  �  L  i  �  m  f  �  �  �  �  �  �  �  2  ��0 ż�`B��`B��C��u�49X�49X�#�
��`B�D�����
���
:�o=P�`<�o��o%   =��m<o<t�<�o<ě�<u<�t�<�1<�1<���<�j<ě�<�`B=D��=\)<�`B<��=+=o=o=\)=t�=e`B=49X=#�
=0 �=0 �=0 �=@�=@�=L��=ix�=y�#=�o=�O�=�+=�C�=�\)=�t�=��P=\=���=�����������������������fcbggkt����~zvtgffff

#)'#








okqt������������wtoo��������������������||�������||||||||||������������������������������������������������������������������������������������������������������������������������HDDJLht���������th[H )5BKNSSPGB5)����������������������������������}������������������NEBCJN[gtu����{tg[NN"#0<@C=<0#"""""""")*-+)
#/37FKG</#
��)5BN[`a[LG5)�����������������������*15BN[^_^[NB<5******		
#%*-.(#
		igjnz�����������zsni  	"/4>AHA;/"
�����


����������RX[gt���tg[RRRRRRRR��������������������")05873)��)15?>55)�
#/<HSMJHB6-#
#,/<HJRSVWQH></$+)./<GHTQH><9/++++++��������������������	
 "####
#%/00/# *56CCKFC60*#��)06751+)�����������������������()696);99;HIRTUTIHG;;;;;;;��������������������mptt���������tmmmmmm�������������������������)7:6)����pt�����������tpppppp��������� ����������#/=@;/�����Tamz������zymkifc]TT		)2313,)��������������������#./3764/,#"./4;@?;9//"��������������������.%'./<HTSMH=</......����������	��������������������������)02,)
���������������������������������������������U�a�n�z�{ÇÓËÇ�z�v�n�a�U�O�N�U�U�U�U�x�~���������������x�x�x�x�x�x�x�x�x�x�x�[�g�t�v�y�u�t�q�l�g�^�[�S�N�L�I�N�Q�[�[���������������������������������������ٹ����ù͹ȹù�����������������������������������
������������������������'�,�3�5�3�.�'�#����!�'�'�'�'�'�'�'�'���������ùùù�������������������������������'�,�4�;�2�'���������������������������������������������������Ѻ��������ɺ��������������������������������(�4�A�k�o�f�D�4���������������ƁƎƚƧ������������ƸƧƎƁ�u�q�n�s�|Ɓù�������������������ùïàÀ�v�vÇÛù�A�N�Z�a�\�Z�N�A�5�2�5�?�A�A�A�A�A�A�A�AE�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E��B�O�[�h�t�z�~�{�u�t�h�[�O�H�B�>�;�;�B�B�r�������������r�r�k�o�r�r�r�r�r�r�r�r�������������������������������������m�������¿ȿĿ������m�T�:�"��"�.�G�[�m�ݿ���$�'�&�"����ݿĿ����������Ŀѿ�Óàèìîùý������ÿùìàÓÌÍÊÓÓ���
�����
��������������������������������(�+�0�(�!����������������������������������������������������;�H�T�a�e�k�p�p�m�a�T�;�"������/�;�U�a�j�n�o�n�n�a�U�T�J�T�U�U�U�U�U�U�U�U�y�������������|�y�o�p�p�y�y�y�y�y�y�y�y�ܹ���������������ܹعйҹܹܹܹܹܹ��"�/�;�K�Q�P�M�H�9�/�"������������ �	�"�s�s�v�v�v�s�r�f�Z�M�M�H�M�M�W�Z�f�r�s�s�s�������������������������������q�g�i�s���)�3�B�H�U�P�N�B�5�)��� ���������������������������������������������(�5�A�N�T�Z�g�k�g�N�A�5�(�"�������(�4�>�A�M�Z�d�Z�M�F�A�4�1�������/�;�;�=�;�0�/�"�!��"�-�/�/�/�/�/�/�/�/�����������������y�m�k�k�m�q�y�������������
�$�0�9�<�:�0�#�����������Ŀ��������ŇŔŠšŧťŦŠŔŇŃ�~�{ŁŇŇŇŇŇŇ�'�,�3�=�3�3�*�'�#�� �%�'�'�'�'�'�'�'�'����%������ ������������������������������������������������������"�.�;�;�D�G�I�G�;�.�"����"�"�"�"�"�"���������������������������������������������"�/�3�5�0�(������������������������������������������������������������������ʾ׾����	��	����׾ʾ����������s�z�������������������s�m�g�^�_�f�r�s������������ŹŭŠśŔŔśŠŭŹ���������e�r�~�������������������~�r�m�f�e�e�e�e�z�������������z�o�n�z�z�z�z�z�z�z�z�z�z����*�C�O�R�O�L�C�A�6�*���	�������#�0�7�<�=�<�;�0�#���
�
���
��#�#�#���������Ƚǽ˽ɽ½���������������������D�EEEEE(E*EEEE D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D|D{D�D�D������������������r�f�Y�P�K�K�O�\�f�r��Y�_�f�k�r�t�v�r�f�Y�W�O�M�K�M�S�Y�Y�Y�Y B f ` J > O E F L V f 8 ) : @ d N $ 9 \ _ ; 7 M B ( ( D ` 3 < Q q - ( \ m I R K I k U 3 A d  | g d n 8 Z J > u : F . f  /  �  Q  1  `  S  j  P  I  m  �  X  �  :  ~  \    �  �    �  �  �    1  3  ^  w  �  �  |  9  6  �  �  m    3  �  �  +  �  g  s  �  �  �    l  �  F  )  Z  }    �  �  �  !  �  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  P  F  <  2  '      	   �   �   �   �   �   �   �   �   �   �   �   �  �  �  �  �  y  \  >    �  �  �  �  ~  s  g  [  G  1      �  �  �  �  �  �  �  �  s  d  U  H  ;  .  !       �   �   �  /  /  /  -  )  $        �  �  �  �  �  _  .  �  �  G   �  :  ?  D  I  M  L  K  K  G  ?  7  /  $      �  �  �  �  �  �  �  �  �  a  :    �  �  �  Z  %  �  �  ~  C    �  �  D  �  �  �  �  �  �  �  �  �  �  p  W  >  %    �  �  �  �  �  �        �  �  �  �  �  �  g  L  0    �  �  �  �  v  V  -  +  (        �  �  �  �  �  �  �  }  m  ]  M  <  *    ?  J  n  t  t  m  ]  :      �  �  �  �  �  �  �  �  �  �  q  r  s  o  b  T  D  3  !  	  �  �  �  q  =    �  �  o  ?  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �              �  �  �  �  �  �  �  �  S  !  �  �  �   �  �  A  �    X  �  �        �  =  W  ;  �  w  �    �  B  V  �  	  	A  	Z  	Y  	A  	  �  �  f    �  T  �  F  |  `    �  B  8  .  #    	  �  �  �  �  �  �  �  �  ~  o  f  _  X  Q  <  J  X  T  E  3    �  �  �  �  �  h  H  )    �  �  �  ~    +  �  �  �  |  *  �  �    :  A  (  �  ?    f  �  
?  �  �  y  o  e  Z  O  E  ;  3  *  "        
            �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  o  O  )    �  �  �  �  �  �  �  �  �  $  �  m  �  �  �  �  �  �  �  �  d  ?  !    �  �  [    �  �  [  i  `  Z  W  I  =  /    �  �  �  �  i  8  �  �  �  �  �  �    �  �  �  �  �  {  _  ?    �  �  q  ;  	  �  �  �  �  �  t  �  �  �  �  �  y  j  U  =  #    �  �  �  {  ?  �  m  �              �  �  �  �  �  Y  !  �  �  y  ?    �  �  A  �  �  �  �  �  �  i  ?    �  �  ,  �  Y  �  k  �  q  N  P  [  c  ]  U  F  4    �  �  �  �  r  N  '     �  �  �  S  �  �  �  �  �  �    r  l  g  h  p  w  z  }  |  x  t  o  k  i  \  X  m  v  s  o  j  b  Y  M  :  #    �  �  �  �  g  =  �    (  C  V  `  c  l  u  j  U  3  
  �  �  +  �  �  +  �  P  E  >  5  u  �  �  �  �  �  �  �  i  <    �  �  r  �   �  �  �  �  �  }  i  U  @  *    �  �  �  �  d  5  �  �  �  N  T  _  _  [  P  ;  "    �  �  �  m  '  �  t    �  8  �  �  �  �  �  �  �  �  �  �  m  M  #  �  �  Q  �  �    �  A  �  �  �  �  k  W  A  '    �  �  �  c  &  �  v    �    �    o  U  8    �  �  �  k  <    �  �  �  n  @    �  �  |  J  �  �  �  �  �  �  �  �  �  �  t  g  Y  K  >  2  &        �  �  �  �  �  �  v  f  V  E  6  )       �   �   �   �   �   �  	@  	�  
  
:  
X  
[  
O  
6  
  	�  	�  	e  	  �  �  �      ^  E  5  @  F  K  M  D  9  +      �  �  �  V    �  �  �  Q  �  �  �  �  �  �  �  �  �  v  i  ^  V  N  H  E  B  ?  =  ;  :  �  �  �  �  �  �  w  i  Z  L  ;  )      �  �  �  �  �  �  K  I  G  E  B  @  >  ;  6  2  -  )  %    	  �  �  �  �  �  ~  x  r  k  `  V  K  @  5  +           �  �  �  �  �  Y  �  �  �  �  v  h  W  A  +      �  �  �  p  :    ^   �   h  L  2    �  �  �  �  �  Y  ,  �  �  �  U    �  s  �  D   �  i  T  ?  (    �  �  �  �  �  r  O  $  �  �  �  [  %  h  �  �  p  ]  I  4    
  �  �  �  �  �  �  �  �  �  �  �  }  l     P  k  W  6    �  �  y  9  �  �  \    �  <  �  -  �  $  A  M  d  S  <  '      �  �  �  �  �  >  �  �  (  �  i  A  x  �  �  �  �  �  X  )  �  �  }  :  �  �  W  �    ,  �  �  �  �  �  �  u  \  C  .    	  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  r  c  S  B  0    	  �  �  �  ]  �  d  �  �  �  �  �  a  F  %    �  �  �  Z  ,  �  �  �  a    �  �  �  s  :    �  �  �  �  �  s  d  [  I     �  �  $  �  �    �  �  R    
�  
�  
:  	�  	�  	H  �  �  2  �  "  �  �    8  U  �  �  �  K    �  U  �  �    �  	  Z  g  f  
\  	  q  �    �  �  �    1  )    �  �  �  N  �  �    Y  
q  	H  �    <  �  t  j  d  E    �  �  i  -  �  �  V    �  X  r    p  �