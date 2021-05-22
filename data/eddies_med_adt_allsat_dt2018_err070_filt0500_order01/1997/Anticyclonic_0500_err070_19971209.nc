CDF       
      obs    :   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?��`A�7L      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M�F   max       P�XB      �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ��P   max       >��      �  |   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?�G�z�   max       @F������     	   d   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�fffff    max       @vhQ��     	  )t   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @/         max       @R�           t  2�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�-        max       @�            �  2�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ��h   max       >y�#      �  3�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�@�   max       B3�      �  4�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�|   max       B4�      �  5�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       ?B
   max       C��      �  6�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       ?A5g   max       C��      �  7�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          �      �  8h   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          I      �  9P   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          E      �  :8   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M�F   max       P��v      �  ;    speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�����+   max       ?�#��w�      �  <   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       �C�   max       >#�
      �  <�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?�G�z�   max       @F������     	  =�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�fffff    max       @vg33334     	  F�   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @.         max       @R�           t  O�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�-        max       @��          �  Pl   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         De   max         De      �  QT   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��s�PH   max       ?���?        R<   	            �         /         M                  
   '      ?            
   
      
   '               >      
   &   7               
   
   4         D   
   
         '   	   �         H   ,N֞^NryWN��~O�F,P�XBOAz�NG�PG�N�GiNV�3P�$�O�AOH��N��$NH��Nы�N~��Px�N�ԊP~�|OZ0�OZ/�N���NX�TN�ĔNYs�N��OZ�	O��O�3OY��N��P:g�NW,N8�P*ĭP$��Ni��O��N��O��N���O��O�}�Nn�N;�UO���M�FNI��N �(O�pO�s�N�oOq��N��N�Z�O�%�Om�Խ�P���
��o�#�
��o��o:�o;��
;��
;�`B;�`B<#�
<49X<D��<u<�o<�C�<�t�<�t�<�t�<�t�<�t�<���<�1<ě�<ě�<���<���<�`B<�`B<�h<�<��<��=+=C�=C�=\)=t�=�w=#�
=T��=]/=aG�=ix�=q��=�hs=��=���=���=�9X=��=ȴ9=�`B=>V>t�>���~������������������&&$()6BDCB>61)&&&&&&@9CNR[got����tg[XNB@'#%$/HRS[aacaZUH<8/'����)Bt���ygB)����2),.57BN[agilg^[NB52��������������������WZTXat�����������h[W���������������������������������������������5BNb]N5 ��������������� �������������
#&.6730#
��UVV^anwz{zyqnaUUUUUU~�������������~~~~~~,(./<EHTU_VUH<//,,,,���������������������������������������	
()34)						CDHUa����������zaJCDCDHUagn{����znaUNHD�������������������������������������������������������������������������#)'024410&#�������


������������������������������������������������������ ���#/<BJPQMH</
�����

��������������������	���������������������������%+/<DHJH</%%%%%%%%%%@:.+43/#
�������
;@������
#"�������������������������������������������������������������������������
""
��69<BO[`e[OJB66666666������������������������
#/<OUUOH</#
��������������������� ���������������
##/4;:/+#
����_\\acgnoonca________

?IIUbbb]UI??????????��������������������{{���������������������������������������������
�����aahmnuz{zysma][\aaaa��������������������zvvz���������������z��������

����������������������������������r�q�r�~������'�(�*�+�'����������������������������������������������������������ʾ׾ؾ��׾ʾ���������������������������B�zĄĆ�q�B������������ìá�����g�t¢�t�n�g�e�\�^�f�g�������ûʻû�����������������������������(�M��������������f�M�4�(�� �������#�/�<�F�E�@�<�/�,�#�����#�#�#�#�#�#čĕĚĜĚĎčā�x�vāĂčččččččč�0�I�nŇŭųŢ�k�U�J���������ĿĥĦ���0�������)�)�.�)�%���� ����������������(�.�2�4�(�#�����������������������������������������������������Ѿ���������������߾۾߾���������������������������������������4�A�M�Z�[�Z�W�M�A�4�(��(�+�4�4�4�4�4�4�����������������������g�Z�J�E�J�Z�����������������������������������������������A�N�Z�p��������������Z�N� �������(�A�z�������������������z�x�m�h�a�_�_�a�j�z�������������	���������������������˺���'�'��������޹ܹӹٹܹ����¿��������¿²¦£¦²´¿¿¿¿¿¿¿¿����!�(�+�!������������������M�Y�f�l�r�����r�f�Y�W�M�G�M�M�M�M�M�M�������������������|�r�o�f�d�f�r�w��àìùÿ������ùìÓÇÇÆ�z�w�w�~ÇÓà��������������������������������ÿ���ŻS�_�l�x�x���������x�l�_�S�R�L�F�C�F�K�S�T�`�m�p�r�n�h�`�T�G�;�.�)�+�0�;�D�G�L�T�<�H�U�a�i�i�a�U�Q�H�<�7�<�<�<�<�<�<�<�<�f������������	��׾����{�f�Z�W�Z�]�f�.�;�G�I�N�G�E�;�7�.�(�&�.�.�.�.�.�.�.�.¥�g�N�5������������������)�[�`�c�`�t�g�����Ŀѿݿ��ݿѿ����m�`�Y�V�Y�`�m����������� ����������������������������%�'�-�'����������������#�/�6�<�H�H�H�<�/�#�#��#�#�#�#�#�#�#�#���������ʾ׾ݾ׾Ӿʾɾ������������������������ƺĺú����������������������������T�G�@�;�5�.�*�,�.�:�;�G�T�[�`�j�k�h�`�T�	��"�/�<�G�J�I�A�/�"��	�����������	Ƴ������������ƳƯƧƳƳƳƳƳƳƳƳƳƳ�G�T�`�a�b�`�T�G�<�C�G�G�G�G�G�G�G�G�G�G���������'�(�#�
��������ĿĳĴĹĿ���ػ����ûʻû�����������������������������������������������������������������ߺ���������������������������m�z���������������z�m�a�^�^�_�`�a�c�m�m�����!�-�1�3�5�5�-�!�����ۺۺ�������������������������������������������D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�����������������żŹŭťťŭŹ����������¿����������¿²ª¦¦²´¿¿¿¿¿¿�������ּ̼����ּܼʼ���������������EuE�E�E�E�E�E�E�E�E�E�E�E�E�E|EsElEiEdEu 1 C � ; J  1 = F + F 0 I 3 [ = S . G D + R Z c : h < = 5 / O � h ' O K - 1 ; W 2 X F 6 e 2 7 � ] l H D f ' I 3 5 V  �  �  X  �  �  �  `  q  �  f     $  �  �  �  �  �  a  �  +  �  �  7  ~  �  �  �  �  6  /  �  }  �  b  x     �  ~  K  R  >  �  ;  �  B  S  �  5  �  ^  m    �  �  �  �  ]  ��h�T���ě�<u>   <�o;�`B=P�`<�1<49X=�1<���<���<�C�<�t�=��<�/=ix�<�1=��
=#�
=0 �=\)<�=C�<�h=C�=�o=@�=8Q�=D��=��=�v�=�P=0 �=�t�=�9X=#�
=aG�=L��=]/=}�=�%=�"�=q��=��>V=��=�v�=�9X=�x�>+=��>y�#>�>��>]/>H�9B)�4B�+B	yB9�B�B�B��B܋B�"B� B2EB�wB$�]Bu�B3�B��B�B��B1�B؅BUB��BB�B 	B%X'B#`�B!�>B�nB�tB�BT�Br�B��B��B	"B��B��B!IRBB$:�B��B!�B��B
B&B(�B�xB��B'YB�EBr�BuB�gA�@�BZ�B�Bt�B)��B��B	"�BAB�B��B��B�&B�oB�B�yB��B$�B��B4�B�iB>�B�:B�B>BB��B6B�B*B 9mB%B?B#}�B"=�B��B��B?bBM B@BC-B��BG]B�ZB:�B!��B@ B$4�B��B!�_B�hB��B6 B?�B@�B_B&��BB�BlB@hB�JA�|BL�B=GBĬ@��[?��4A�;�AMd�A�uA�Kf@�_�A>��AnA�rA꬞A�h?A2RA��tAŮAҷCA:�*A�W�A���A�
�A��A�-�?B
A�7�@a�a@ݠ�@�r�A�y�A�}@�K�AeӡA�,JALP�Ac�A���A�{Ar��A���@��A»�AL��@%��AeՉA��B��Af�4A��@���A��@N-JA��&@Z�A��?C���A�#�A��*@��C��@�$?��A�r�AK�A�rZA���@�A?�hA�z�A�HA�y9A�aoA1�gA��AV�FAҁ�A:��A�{�A��A��mA���A��"?A5gA�U;@c��@��@�M�A�Z?AЇ"@��EAf�4A�bAM VAb��A���A�sqAr��A�x�@�"�A�sWAL��@#��AdkA��BU�AeUmA�t�@�m�A���@L�A�y&@e'�A�v&C���A���A��@��C��   	            �         /         N                     (      ?            
         
   '            	   ?         '   7                  
   5         E   
            '   	   �         I   -               I         1         G                     %      7                                       5         /   )                              !                                                3         '         E                           /                                       -         /                                                                  NQ+�NryWN��~O%PO8�O`LNG�P�NFJNV�3P��vN��OH��NP�eNH��Nw3�N~��O�IN�ԊPK�O#�TOM&	N���NX�TN�Z�N �Nm�NO0G)NZ��O�3OY��N��PF6NW,N8�P*ĭO'�INi��N��\N��N��eNA�>O��O�}Nn�N;�UO\
%M�FNI��N �(O�pO�s�N�oO+�N��N�Z�O�%�O*ڌ  �  $  �  �  ?  m  �  9  �  E    Q  �  �  �  �  �  �  C  -  �  v  
  �  �    �  *  8  :  ]  �    �  F  _  C  j    �    �  �    <  H  
z  �  \  b  �  �  �  �  �  �  `  
"�C����
��o��o=49X��o:�o<�C�<49X;�`B<o<T��<49X<e`B<u<�1<�C�<���<�t�<�/<�9X<���<���<�1<���<���<�/<��=\)<�`B<�h<�='�<��=+=C�=�+=\)=�P=�w=,1=]/=]/=q��=ix�=q��=�E�=��=���=���=�9X=��=ȴ9>I�=>V>t�>#�
��������������������&&$()6BDCB>61)&&&&&&@9CNR[got����tg[XNB@5014:<GHUU[]^\UH<<55���)5B[iqqhZTD)���-/15;BNP[]fg[XNB85--��������������������c]]bht������������nc��������������������������������������������$5ANa\N5�����������������������������
#&.6730#
��[[[ahnpvunia[[[[[[[[~�������������~~~~~~4//<HMTJH<4444444444����������������������������������������	
()34)						QIIUan����������znaQFHHQU]anvz��}znaUOHF�������������������������������������������������������������������������#+,013310$#�����	���������������������������������������������������������� ���#/<BJPQMH</
�����

������������������������������������������������%+/<DHJH</%%%%%%%%%%@:.+43/#
�������
;@�����
������������������������������������������������������������������������

�����B;>BO[]b[OBBBBBBBBBB����������������������#/<HMSSMH</#
��������������������� �����������������
#(,.//&
��_\\acgnoonca________

?IIUbbb]UI??????????��������������������{{��������������������������������������������

�������aahmnuz{zysma][\aaaa��������������������zvvz���������������z���������

������ɼ���������������������z�������������������'�(�*�+�'������������������������������������������������������������ʾ׾پ׾׾ʾʾ�����������������������1�i�u�w�u�h�[�O�B�6�)�������������t�t�i�g�_�a�g�j�t�t�������ûʻû����������������������������4�M�s���������������s�Z�A�4����
��4�/�<�=�@�<�<�/�#�"�!�#�'�/�/�/�/�/�/�/�/čĕĚĜĚĎčā�x�vāĂčččččččč�
�0�I�nűŠŔ�i�U�I���������Ŀħĳ���
������&�)�-�)�����������������������(�.�2�4�(�#�����������������������������������������������������Ѿ���������������߾۾߾������������������������������������������4�A�M�Z�[�Z�W�M�A�4�(��(�+�4�4�4�4�4�4���������������������������Z�U�Q�T�d�s�������������������������������������������(�A�N�g�v�����������Z�N�'�������(�����������������������z�m�k�e�d�g�m�z����������� ����	���������������������˺���'�'��������޹ܹӹٹܹ����¿��������¿²¦£¦²´¿¿¿¿¿¿¿¿����!�&�(�!������������������M�Y�f�j�r�����r�f�Y�X�M�J�M�M�M�M�M�M�������������u�s�~����������àìù����������ùìàÓÇ�~�{ÁÇÓÝà���������������������������������������һS�_�l�x�x���������x�l�_�S�R�L�F�C�F�K�S�T�`�m�p�r�n�h�`�T�G�;�.�)�+�0�;�D�G�L�T�<�H�U�a�i�i�a�U�Q�H�<�7�<�<�<�<�<�<�<�<������������ݾ�����׾������o�h�c�f�n��.�;�G�I�N�G�E�;�7�.�(�&�.�.�.�.�.�.�.�.¥�g�N�5������������������)�[�`�c�`�t�g���������ÿ����������������|�|���������������� ������������������������������%�'�,�'����	������������#�/�6�<�H�H�H�<�/�#�#��#�#�#�#�#�#�#�#���������ʾʾоʾž����������������������������ú��������������������������������T�G�@�;�5�.�*�,�.�:�;�G�T�[�`�j�k�h�`�T��"�/�;�E�H�G�?�4�/�"��	���������	�Ƴ������������ƳƯƧƳƳƳƳƳƳƳƳƳƳ�G�T�`�a�b�`�T�G�<�C�G�G�G�G�G�G�G�G�G�G�����������������
�����������������̻����ûʻû�����������������������������������������������������������������ߺ���������������������������m�z���������������z�m�a�^�^�_�`�a�c�m�m�����!�-�1�3�5�5�-�!�����ۺۺ�������������������������������������������D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�����������������żŹŭťťŭŹ����������¿����������¿²ª¦¦²´¿¿¿¿¿¿�������ּ̼����ּܼʼ���������������E�E�E�E�E�E�E�E�E�E�E�E�E�E�EEwEtEuEzE� 1 C � 8 Z  1 C E + B & I : [ - S 1 G 6 / P Z c 7 m F 5 . / O � g ' O K # 1 8 W % N F 0 e 2 3 � ] l H D f   I 3 5 N  a  �  X  I  �  H  `    m  f  �  �  �  q  �  |  �  �  �  ]  f  �  7  ~  �  �  �  ~  p  /  �  }  �  b  x     _  ~  !  R  �  l  ;  ]  B  S  �  5  �  ^  m    �  e  �  �  ]  �  De  De  De  De  De  De  De  De  De  De  De  De  De  De  De  De  De  De  De  De  De  De  De  De  De  De  De  De  De  De  De  De  De  De  De  De  De  De  De  De  De  De  De  De  De  De  De  De  De  De  De  De  De  De  De  De  De  De  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  ~  f  P  >  +  $          �  �  �  �  �  �  �  �  �  �  �  p  V  :    �  �  v  �  �  �  �  �  �  }  o  a  S  E  6  &    �      �  �  �  �  �  �  �  �  �  �  �  m  I    �  l     x  �    
@  
�  
�    �  �  (  ?  2    �  K  
�  
)  	j  �  �  L  �  q  b  i  k  l  j  e  \  N  <  $    �  �  �  o  =  �  �  G  ?  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  |  t  l  d  \  �  �  �  '  6  9  1      �  �  �  �  \    �  Q  �    ]  @  U  h  z  �  �  �  �  �  �  �  q  L    �  �  i  $  �  G  E  7  *        �  �  �  �  �  �  �  �  �  �  �  �  �  �      �  �  �  u  ;    �  �  i  8  (  �  z  *  �    i  7    '  5  F  L  =  (    �  �  �  y  K    �  _  �  t  �  U  �  �  �  }  x  q  c  R  A  0      �  �  �  �  K    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �    {  x  t  q  j  b  Z  Q  I  A  8  0  '          �  �  �  �  �  �  �  �  z  _  ?    �  �  }  I    �  �  �  �  �  �  �  �  �  o  ^  Q  J  a  �  �  �  �  �  �  �  �  �  T  t  �  �  �  �  �  p  M  $  �  �  ~  +  �  r    �  �   �  C  =  7  1  +  &              �  �  �  �  �  �  �  �  t  �    '  ,    �  �  �  l  3     �  �  �  8  �  <  �  r    �  �  �  �  �  �  �  �  {  \  8    �  �  U    �  �  `  ^  t  u  o  _  O  =  *  0    �  �  �  ^    �  �  T  �  �    
  �  �  �  �  �  �  �    e  J  .    �  �  �  �  s  N    �  �          �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  q  X  4       �  �  �  �  �  �  t  S             �  �  �  �  �  �  �  �  �  ~  Y  2  	   �   �  \  l  }  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �    (  *  $      �  �  �  l  !  �  u  !  �  f  E  A  �       "  *  0  5  8  8  6  &    �  �  �  �  R    �  �  ;  :  1  )      
    �  �  �  �  �  �  �  u  )  �  B  �  K  ]  Q  A  /    �  �  �  �  �  d  ;    �  q    �    2  .  �  �    )         �  �  �  �  �  v  _  G  -      V  �  �  �        �  �  �  �  �  �  s    �  '  �  �  -  �   �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  l  T  ?  5  ,  "  F  G  H  I  J  J  J  H  D  =  5  +      �  �  �  �  �  �  _  6    �  �  �  �  }  x  f  C    �  �  �  Q  �  �    �    l  �  �    :  A  4  !  (  :  B  8    �  �    �  �  �  j  \  N  @  3  (        
      _  �  �  %  k  �  �    �    �  �  �  �  �  h  J  6  &    �  �  �  ]  1  O  Y  d  �  y  f  H  (    �  �  �  �  ^  :    �  �  �  �  �  �  �            �  �  �  �  �  j  J  (    �  �  �  P  �  e    �  �  �  �  �  �  �  x  f  R  ;  #    �  �  �  �  �  �  �  �  �  �  �  �  �  �  ~  r  e  X  K  :       �  �  �  �  �    
  �  �  �  �  �  c  :    �  �  1  �  e  �    �    <  9  5  2  .  +  '  $         �  �  �  �  �  �  �  v  b  H  B  <  6  4  2  0  '        �  �  �  �  �  �  �  �  �  	w  	�  
  
D  
d  
z  
o  
[  
;  
  	�  	�  	H  �  l  �  �  �  a  x  �  �  �  �  �  �  �  �  v  ]  E  2  %      %  y  �  ,  �  \  D  ,      �  �  �  �  �  �  �  �  �  �  �  �  �  p  Y  b  :    �  �  �  �  y  ]  =    �  �  �  �  �  �  w  a  J  �  `  8    �  �  `     �  �  G  �  �  �  �  a    �    �  �  �  �  q  I    �  �  \    �  �  M    �  v    p  �  �  �  s  b  J  .    �  �  �  �  �  `  ?    �  �  r  (  �  �  W  �  �  �  �  �  �  [    �  #  �  �  �  �  �  !  	�  !  ;  �  �  j  I  )    �  �    T  5    �  �  �  �  �  �  �  �  �  �  �  |  b  L  /    �  �  �  �  f  <    �  �  �    �  `    �  �  �  i  )  �  �  0  
�  
s  
  	|  �  >  �  �  &  q  
  
  
  
   
!  
	  	�  	�  	_  	  �  Y  �  �    s  �  9  �  R