CDF       
      obs    8   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?pbM���   max       ?���vȴ:      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M�x   max       P��      �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       �u   max       >o      �  l   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>��G�{   max       @E�=p��
     �   L   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�z�G�    max       @vs\(�     �  )   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @0         max       @P            p  1�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @ˣ        max       @�;           �  2<   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �D��   max       >/�      �  3   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A��   max       B,b      �  3�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�wa   max       B+�E      �  4�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       ?H�   max       C�iJ      �  5�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       ?��   max       C�g�      �  6�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          i      �  7|   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          A      �  8\   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          !      �  9<   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M�x   max       O�J�      �  :   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?����u�   max       ?��\��N<      �  :�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       �u   max       >o      �  ;�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>������   max       @E��
=p�     �  <�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�z�G�    max       @vs\(�     �  E|   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @)         max       @N            p  N<   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @ˣ        max       @�@          �  N�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         B�   max         B�      �  O�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�쿱[W?   max       ?���#��x     �  Pl                     9               	         *          *               
   	      h            
      +   \         "               R            5                  #      D         O���NŸ�NU+N�� NLEaN�O�KDN��O.rNL-O��qO�OzO�NX��OK�OO�ZO-͈O��O1>N��O���N�$gN��O; 8P��ORA+NKÒOòOB:NdfP(@O�'N+U<M�xO��_O��OdHO8,OV�OO�PAN�C�NʫjO�PO���O�@O�_Ok�N���N�EOBXxN�luO�E�N�юN=�MN��.�u�49X��`B��o;D��;��
;�`B<o<#�
<#�
<D��<u<u<�o<�o<�t�<�t�<�1<�j<���<�/<�/<�/<��=o=+=C�=\)=�P=#�
='�='�=<j=P�`=P�`=P�`=T��=Y�=Y�=aG�=aG�=aG�=e`B=m�h=u=}�=}�=��=�t�=��w=�-=�v�=��=��#>o>oGHOUanz��������znaQGVUZ[fgitvx}xtga[VVVV67BGMN[\[[SNGB666666!)05BNQNJE>53)����������������������������������������NMQ[t����������tg[UN����� �����"/:;FB;4/%"	��������������������86104<HUVakprmfaUH=8_agt�������������tq_����������������������������������������������������������
#$/0<=??=/*#!,**.<Han����z{aUH</,��������������������!#07<IMUVROI<0/(#TY[ht����������wtihT��������������IEFIO[ht{{~xth[WOI��������������������
#%(*#
��������

��������������287��������������	���������������������������������������������������������������������������������������������
#/32/#
����������
##
�������������������������  )6BOUQOE<.) ������������^[gqt������������tg^#/<>HHJKH</#�����������������������)-..21/)	���025:BLN[\ghhg_[NB500 #0<ISTIA<0#��������PP[`bgt����������t[P����))3.)�������������������������}z}����������������}&(),5;BBDCB555+)&&&&��������������������`_aafmwz�������zmha`npuz|�������{zzpnnnn���}���������������#
 ����
#$)$###����������������������������


�������E�E�E�E�E�FE�E�E�E�E�E�E�E�E�E�E�E�E�E�������������������������������������������)�.�5�B�E�E�B�5�4�)� ���������G�T�U�`�i�m�o�p�q�m�h�`�T�G�<�;�6�;�<�GŹ������������ŹųŰŹŹŹŹŹŹŹŹŹŹāčĔčĉā�t�h�c�h�t�~āāāāāāāāÇÓàëïîêàØÓÏÇ�z�n�f�]�_�f�nÇ�S�_�l�x�����x�l�_�T�S�N�F�D�F�I�S�S�S�S�H�T�a�m�s�z����z�y�m�a�Y�T�O�H�E�C�D�H���������������������������������������������������������������������s�o�i�g�s�������������������������s�q�j�d�f�m�s����������ɼʼͼ̼ʼ����������������������������ƺǺ������������������������������ûлܻ����ܻջû�����������������������&�(�4�5�A�A�,�(������������������������
��	�������������������ìùÿ��������ùóìàÓËÇÇËÓÕàì�������������������������r�f�e�f�r�x��H�U�X�a�Z�Y�[�U�O�<�/�)�%�$�)�/�3�<�>�H�/�;�H�Q�T�V�T�H�;�/�)�%�/�/�/�/�/�/�/�/������������ɾľ����������s�j�]�b�f�s��ʾ׾ݾ������������׾ӾʾʾɾȾʾʿT�`�m�o�m�l�`�W�T�G�;�3�;�;�G�L�T�T�T�T�<�H�U�a�n�n�i�g�a�U�M�<�/�'�#���#�)�<�<�H�nÇì����F�N�A����ùÓ�n�^�J�:�<���	��"�.�;�F�I�C�;�.�"��	��������5�;�?�5�)�(�������(�1�5�5�5�5�5�5�ùϹܹ��� ���������ݹϹ͹ùù����������������������������������t�t�x�����a�m�t�z�{�z�x�m�i�a�X�X�a�a�a�a�a�a�a�a�Z�f��������������վ������s�R�4�7�M�ZD�D�D�D�D�D�D�ED�D�D�D�D�D�D�D�D�D�D�D����������������������������������������'�3�:�3�0�'��������������Y�r�����������º����������r�Y�G�<�B�B�Y�����������������������������������������B�O�U�[�]�h�m�n�l�h�[�B�6�1�0�4�6�<�?�B�B�D�N�[�_�b�^�[�N�N�B�5�/�)�,�0�5�>�B�B�������������������������}����������������#�0�7�A�D�E�<�0�#��������ĽĻĿ������I�U�Y�b�h�n�o�o�n�c�b�U�S�I�F�C�E�G�I�I�ֺ�����������ںֺѺκϺպֺֺֺ�¿������������������¿²ª¦£¥¦«²¿�ܻ� ���'�9�?�?�4�'������ܻӻͻ׻ܾM�Z�f�m�o�o�k�f�f�Z�M�E�C�E�I�M�M�M�M�M�����������ĽννĽ����������u�s�l�c�p����������������
�	���������������������h�u�xƁƌƁ�v�u�h�\�U�O�O�O�\�a�h�h�h�h�-�:�?�F�T�\�_�b�_�S�F�;�:�-�$�!��!�(�-ŠŭŷŹ��������źŭũŠŔŇłŀńŇŔŠ�ּ������ �����ּʼƼ������ʼֻּּּ�����@�M�T�_�b�Y�@�'�����������a�U�T�U�Y�b�e�n�{ŃŇŊŌŉŇ�{�n�a�a�aE7ECEPERE\E`E\EPECE7E3E5E7E7E7E7E7E7E7E7EiEuE�E�E�E�E�E�E�E�E�E�E�EyEuEpEiE`EiEi C . v T 2 x   h L ^ 8 / I ? P - m / 1 o 2 4 O c + � ; i ) & = ^ 4 K > ^ B W # = O I k A B [ + ` I O - A 0 ' Q R  -  �  �  B  T  F  3  �  |        i  w  �  Y  �  w  P  �  �  �  ,  �  �  �  �  �  +  �  �  �  �  N      2  �  5  �  7    /  Z  ~  a  \  N  �    �  �      l  �<t�:�o�D��<t�<49X<t�=�%<u=C�<�C�=�P<�j=�w<�j=ix�=+=L��=}�=�P=<j=C�=@�=t�=�w=L��>I�=]/=�w=D��=L��=aG�=���>O�=e`B=aG�=�{=���=��T=�o=�1>C�=��
=�O�=��T=�l�=��w=�E�=��-=��=�{=���=�/>/�>��>\)>�uBDB	WrB��B�B�B"xB	��B�&A��B:6B�B
��B "�B"m�B"�gB�*B�B!�[B&�B[tBt�B�B gBH�B��B�hB@9B�aB L�Bs�B�B�VBHxB SB�%BB�B
��BK2B]�B�=B6B%��B�Br*B�aB+�0B�BQ%B,bA�x�B�B�BB��B�dB��B	?�B@B'�B�bB?�B	��B�bA�waB��B�JB 3B 2lB"OCB"��B�
B>%B"?B&�BF%B��B�B �B/�B�aB��B=�B��B AABA�B>�B!HB@TB�4B̄B6�B@�B�BH�BA�BpBA�B%�B?�B��B��B+�=B!B?�B+�EA��[BF�B<�B=	B��B6�C�iJA�/BA��AfۺA�dA��Aɔ�@���A��NA�k�A�u,AG@�(�@'%�@�$A�4�A�q�A�U>@�A�RA��JAH;�AT�Ag�MA��Aҁ�A]�,A�Eu?H�A�9)A�(�AFȄC��MAHV�?���@��A�D�A�dA���A���A�z1A@Cc+A�tH@�>�A?�A L�A��<BX/@���A���A>X@��}A��C���C��@C�g�A��A�zaAgWA�c�Aܓ4A�N�@� ?A�~�AАXA���AF�@�Q@$~o@���A��tA�'A�w!@愾AÀ`A�}�AGAS�NAhA��A�qdA_��A���?��A���A�~'AE��C��JAH��?�Y|@��A�w�A���A�wA�Q�A��A@D&�A���@�A>�%A D�A�6�B"�@{��A�}�Ax
@��AA�_C���C��               	      9               	         *      !   *               
   	      i                  +   ]         #               R            6                  #      E                                                                                       A                  +            %               !                                    #                                                                                                         !                                                                        O���NŸ�NU+N��NLEaN�O0��N��N��;NL-O'E"O�OzNnۅNX��O1�wONn��N���O��N��XN��O���N�K�N'v�O)&�O�YO3�.NKÒOòOB:NdfO��O��N+U<M�xO'KWO��OdHO8,O
��O�v�NϯONʫjO�PO���O�@OSz�Ok�N���N�EOBXxN�luO�J�N�юN=�MN��.  �  B  '  �  �  G  	�  �  �  �  U  F  b  K  �  �  �  j    T  �  �  �  n  �  R  H  `  �  �  �  (  �  �  ~  �    V  A    8  �  �  �  �  �  �  �  |  C  L  \  
c  �  �  ��u�49X��`B��o;D��;��
<ě�<o<�C�<#�
<���<u<���<�o<���<�t�=C�<�`B<�j=o<�/<�/<�`B=C�=+=���=t�=\)=�P=#�
='�=T��=y�#=P�`=P�`=��=T��=Y�=Y�=}�=�7L=e`B=e`B=m�h=u=}�=�\)=��=�t�=��w=�-=�v�=�/=��#>o>oGHOUanz��������znaQGVUZ[fgitvx}xtga[VVVV67BGMN[\[[SNGB666666$)5BDBB;5,)%����������������������������������������YVV[_gt��������tg_[Y����� �����	"/;@=;/,"	��������������������;667<HJU_hjeaYUHG@<;_agt�������������tq_������������������������������������������������������������
#$/0<=??=/*#!536<HQUUUSH<55555555��������������������!#07<IMUVROI<0/(#z|������������zzzzzz��������������IEFIO[ht{{~xth[WOI��������������������	
!###
								��������

�������������
������������������������������������������������������������������������������������������������������
#'--*#
���������
 
���������������������������
#)=<;96*)
������������^[gqt������������tg^#/<>HHJKH</#�������������������������(-//.,)'435;BMN[[gghg^[NB544 #0<ISTIA<0#��������PP[`bgt����������t[P����))3.)�������������������������}z}����������������}&(),5;BBDCB555+)&&&&��������������������`_aafmwz�������zmha`npuz|�������{zzpnnnn�������������������#
 ����
#$)$###����������������������������


�������E�E�E�E�E�FE�E�E�E�E�E�E�E�E�E�E�E�E�E�������������������������������������������)�.�5�B�E�E�B�5�4�)� ���������G�I�T�`�e�k�i�`�T�S�G�@�;�9�;�F�G�G�G�GŹ������������ŹųŰŹŹŹŹŹŹŹŹŹŹāčĔčĉā�t�h�c�h�t�~āāāāāāāā�zÇÓÜàãçåàÚÓÇÁ�z�n�e�g�n�u�z�S�_�l�x�����x�l�_�T�S�N�F�D�F�I�S�S�S�S�a�b�m�m�v�x�s�m�a�_�V�T�M�K�T�Z�a�a�a�a�����������������������������������������������������������������y�s�q�r�s�����������������������������s�q�j�d�f�m�s��������¼ü������������������������������������ƺǺ����������������������������������ûлܻ����ܻӻû�������������������&�(�4�5�A�A�,�(��������������������������������������������������àìù����������ùììàÓÐËÐÓÜàà�������������������������r�f�e�f�r�x��<�H�K�Q�I�H�<�4�/�-�*�)�/�4�<�<�<�<�<�<�/�;�H�Q�T�V�T�H�;�/�)�%�/�/�/�/�/�/�/�/������������ɾľ����������s�j�]�b�f�s��ʾ׾ھ����������׾վ˾ʾʾʾʾʾʾʿT�`�k�h�`�T�H�G�F�E�G�R�T�T�T�T�T�T�T�T�/�<�H�U�a�a�g�f�a�U�J�<�/�)�#���#�*�/���������)�3�6�/�)��������úö�����޿	��"�.�;�D�G�@�;�.�#���	���������	�5�;�?�5�)�(�������(�1�5�5�5�5�5�5�ùϹܹ��� ���������ݹϹ͹ùù����������������������������������t�t�x�����a�m�t�z�{�z�x�m�i�a�X�X�a�a�a�a�a�a�a�a����������ʾϾȾ��������s�`�J�G�Q�Z�f�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�Dƾ��������������������������������������'�3�:�3�0�'��������������e�r�~�������������������~�r�e�d�Y�W�\�e�����������������������������������������B�O�U�[�]�h�m�n�l�h�[�B�6�1�0�4�6�<�?�B�B�D�N�[�_�b�^�[�N�N�B�5�/�)�,�0�5�>�B�B�����������������������������������������������
��#�/�9�;�0��������������������I�U�X�b�g�n�o�n�n�b�b�U�T�I�G�D�E�H�I�I�ֺ�����������ںֺѺκϺպֺֺֺ�¿������������������¿²ª¦£¥¦«²¿�ܻ� ���'�9�?�?�4�'������ܻӻͻ׻ܾM�Z�f�m�o�o�k�f�f�Z�M�E�C�E�I�M�M�M�M�M���������ýý������������z�y�w�v�y��������������������
�	���������������������h�u�xƁƌƁ�v�u�h�\�U�O�O�O�\�a�h�h�h�h�-�:�?�F�T�\�_�b�_�S�F�;�:�-�$�!��!�(�-ŠŭŷŹ��������źŭũŠŔŇłŀńŇŔŠ�ּ������ �����ּʼƼ������ʼּּּּ��@�M�Q�]�_�W�@�8�'������������a�U�T�U�Y�b�e�n�{ŃŇŊŌŉŇ�{�n�a�a�aE7ECEPERE\E`E\EPECE7E3E5E7E7E7E7E7E7E7E7EiEuE�E�E�E�E�E�E�E�E�E�E�EyEuEpEiE`EiEi C . v @ 2 x  h S ^ > / 4 ? H -  5 1 K 2 4 L _ & K 7 i ) & = M 0 K > E B W # 2 Q G k A B [ % ` I O - A % ' Q R  -  �  �  �  T  F  u  �      x    |  w  �  Y  y    P  �  �  �  �  N  i  �  �  �  +  �  �  �  %  N    k  2  �  5  6  �    /  Z  ~  a  �  N  �    �  �  �    l  �  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  �  �  �  �  �  �  a  4     �  �  v  W  1    �  |    �  M  B  3  $            �  �  �  �  �  �  �  �  t  R  0    '      �  �  �  �  �  �  �  �  u  [  B  (    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  n  S  6    �  �  �  �  o  �  �  �  �  �  �  �  �  �  �  e  A    �  �  ~  A  �  �  @  G  <  1  &                �  �  �  �  �  w  Y  ;    	  	N  	p  	�  	�  	�  	�  	�  	  	L  	
  �  I  �  B  �  �  %  ^  �  �  �  �  �  �  �  w  k  _  Q  B  3      �  �  �    �  
  b  �  �  �  �  �  �  �  �  �  �  �  \    �  Z  �  V  �  �  �  �  �  �  �  �  �  �  {  [  :    �  �  �  q  F     �   �  $  2  @  I  Q  T  R  H  =  /  !      �  �  l    �  g  �  F  >  6  +        �  �  �  �  �  �  ~  q  \  F  2  "    �  �  �    !  4  F  W  a  b  `  W  G  )  �  �  !  {  �  �  K  G  C  ?  9  4  ,  "    
  �  �  �  �  �  V     �  �  h  �  �  �  �  �  �  �  �  m  H  (    �  �  L  �    h  �  i  �  �  �  �  �  �  �  �  �  �  j  M  -    �  �  *  �  P   �    7  "  R  Z  U  N  K  h  �  �  �  z  ]  @    �  �  K  �  1  T  f  j  g  _  N  2    �  �  [    �  c    �  =  �  K    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  r  h  h  v  a  j  �  )  C  N  S  S  K  7    �  �  �  i  ,  �  �    Z  �  �  �  x  m  b  V  K  @  3  &      �  �  �  �  �  �  r  �  �  �  z  q  h  ]  P  ?  (  
  �  �  �  m  H  &    �    �  �  �  �  �  �  �  �  �  �  �  �  u  b  M  9  #    �  �  b  [  S  M  H  H  Z  l  o  o  h  [  N  ?  0  #          �  �  �  i  J  +    �  �  �  �  �  �  �  n  L  )    �  �  @  �  �  	�  
f  
�    2  P  K  &  
�  
r  	�  	^  �  �  @  �    8  C  G  ?  /    
  �  �  �  �  n  B    �  �  R    �  t  `  M  :  (      �  �  �  �  �  �  �  �  s  ^  G  1      �  �  �  i  Q  A    �  �  �  �  f  B     �  �  �  a  '   �  �  �  �  w  Y  4    �  �  �  �  �  �  �  �  �  t  H     �  �  }  k  S  3  
  �  �  �  i  B    �  �  �  �  S    �  �  �  �  �    (  (    
  �  �  �  �  \    �  K  �  L  �  �  �  f  �  �  �  f  $  �  l  �  {  �  i  
�  
*  	>    E  !  �  �  �  }  h  T  D  5  %    �  �  �  �  �  �  y  a  G  -    ~  �  �  �  �  �  }  t  j  a  P  9  "    �  �  �  �  t  S  g  t  l  k  r  w  y  �  �  _  .  �  �  Y  �  �    �  �  �        �  �  �  �  �  �  s  O  '  �  �  y  '  �  I  �   �  V  @  ,      �      �  �  �  �  [  $  �  �  G  �  �  z  A  :  3  +  #        �  �  �  �  �  h  C    �  �  �  �                   �  �  �  �  W    �  �  ,  �  �  �  �  �  &  7  4  '    �  �  �  f  �  O  �  
�  
3  	]  �  �  i  �  �  �  �  �  �  h  :    �  �  @  �  �  ]  �  �  
  �  !  �  �  �  �  }  c  H  .    �  �  �  L    �  q  ,  �  �  �  �  �  �  }  d  C    �  �  �  W    �  �  U    �  Y  �  g  �  �  �  �  �  y  X  *  �  �  P           �  l  �  '  s  �  �  �  �  �  �  �    m  V  6    �  �  )  �  f  �  ~   �  j  c  |  �  �  �  �  �  �  [  )  �  �  p  #  �  s  �  n   �  �  �  �  �  �  �  �  �  �  �  |  l  T  =  +      �  �  �  |  r  h  Y  J  5    �  �  �  �  g  ?    �  �  �  d    �  C  A  >  :  3  -  %      �  �  �  �  �  �  i  O  H  I  J  L  C  =  1      �  �  �  e  0  �  �  _  �  �  �  $  �  u  \  F  /       �  �  �  �  q  Q  /  	  �  �  �  (  �    4  	�  
b  
_  
F  
%  	�  	�  	�  	i  	!  �  w    �  T  �  .  `  �  a  �  Q  '    �  �  �  ]    �  �  4  �  �  ;  �  |    �  �  �  �  e  0      �  �  �  �  �    e  J  .    �  �  �  �  �    d  H  #  �  �  �  �  b    �  j    �  b  �  �  �  �