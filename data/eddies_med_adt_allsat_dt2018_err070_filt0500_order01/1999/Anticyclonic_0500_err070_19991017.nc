CDF       
      obs    4   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?Ӆ�Q�      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M�u�   max       P���      �  |   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       �\)   max       =�G�      �  L   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>ٙ����   max       @E�ffffg            effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�Q���   max       @v������        (<   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @+         max       @P            h  0\   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�9        max       @��`          �  0�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ��h   max       >�Q�      �  1�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A��z   max       B,�)      �  2d   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A���   max       B,�D      �  34   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       ?�r�   max       C��6      �  4   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       ?�I   max       C���      �  4�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max         1      �  5�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          ;      �  6t   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          1      �  7D   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M�u�   max       P:�!      �  8   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��u%F   max       ?瞃�%��      �  8�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       �\)   max       >/�      �  9�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>޸Q�   max       @E�(�\        :�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�Q���   max       @v��G�{        B�   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @!         max       @P            h  J�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�9        max       @�           �  K,   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         G
   max         G
      �  K�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?������   max       ?��ߤ@     @  L�      
                         
       &   P   �            2      6   
   !         %      1            H   V                  -        1      
   m   	         %   <         N��O�NO���N�F�N 6�N���OZ�"O%d�OTDNS��P8d�O�E"P[g@P��Nݎ�OA=CN456O�ǅOq��P��N�dIO%�O�JO	O�\Om�O��M�u�N
�OY%Oz�NP���O���Np[ N��8N\��O���P'�N�~�PC�PZ�PM���NFqO���N|O��OmvOʸ�O��OE]DN6��N3D�\)��`B�D���D���o<t�<T��<u<u<u<�o<�1<�9X<�9X<�j<���<�/<�h<�h<��<��=o=o=+=t�=�w=#�
=,1=,1=49X=<j=<j=<j=D��=D��=H�9=L��=L��=T��=]/=e`B=e`B=��=�C�=�C�=�O�=�t�=�t�=�^5=\=��=�G���������������������546:<@HUaahdaaWUH<55"'/;AEKLQPJH;3(�����


��������������� ��������������������������������������������������������������/<HU`ihaUH</#
��������������������#/a��������zaH<4+��%07<>23)����������5:<2#
�����NUWan����������zn]UN).45855)��������������������x~����������xxxxxxxx�{z�����������������a_`et������������thaedht��������������te����������������������������������������st|��������������tts���������

�����������������������������������������������������������������,//<>HKH</,,,,,,,,,,����������������������������

������������
(#
�������������)BD6
����������"'*&��������

���������		
#/0/,$#
				�������������������� ��)47BXWRN5) c\aisxw~����������tc�������������������������������������������g[NB8.'*2B[t�����WR[[hphh][WWWWWWWWWWU[`hptxzvtihb[UUUUUU���������
�����%)6BMEB6-)%%%%%%%%%%�����������������������������

��������),..)����������"!	�����xuz����������������x

������������󺰺����������������������������������������������������������������������~�z�����T�a�m�x�z�����z�m�a�T�;�/�"���"�;�H�T�A�N�T�Z�f�Z�Z�N�A�6�5�1�5�9�A�A�A�A�A�A���������������������}������������������/�<�H�L�Q�U�X�U�H�<�/�-�%�$�/�/�/�/�/�/�����!�-�:�F�N�N�F�:�-�!�������������*�6�8�:�6�0�*������������������������������������������ ��n�z��z�r�zÇÈÇ�z�q�n�e�e�h�j�n�n�n�n��"�(�������������������������������������¼¼����������f�Y�M�R�Y�r�|�������5�B�N�g��t�k�W�B�5�(����������������5FoFuF|F�F�F�F~FqFcFJF=F1FE�E�FFFJFbFo����������������������������������������ù����������ùìÓÇ�~ÂÇÓÓÐÓàìù�T�`�c�d�f�`�Y�T�G�E�G�I�T�T�T�T�T�T�T�T���������������������ùìéâÚÛàì�ž�4�A�M�^�\�T�M�A�4�(���������������-�<�<�8�8�1�"��	���������������𻷻ûлٻѻлȻû������������������������4�@�M�Y�]�f�r�|�����r�f�Y�M�@�7�*�+�4�<�<�H�M�U�W�U�T�H�<�/�%�#���#�#�/�5�<�A�M�Z�f�s�|�����������s�b�M�E�A�4�0�A�Ľн�����������Ľ������|�������ļ��&�'�+�)�����������������������лܻ�����ܻлû���������������������
�������������
����
����������
�
�
�
�
�
�
�
�
�
�������ʾ׾߾�׾����������}�q�x�������E�E�E�E�E�E�E�E�E�E�E�E|EuEqEqEuE{E�E�E���������̺ĺ��������w�b�L�?�<�C�M�~���.�;�G�T�\�]�Y�T�P�G�.�"��	�����"�.��"�.�;�<�A�;�.�"������������T�`�m�m�r�m�h�a�`�_�T�G�E�C�G�O�T�T�T�T�/�;�H�P�R�H�D�;�6�0�/�.�/�/�/�/�/�/�/�/�y�������������������y�m�`�[�P�O�T�`�m�y�m�y�������ѿ������￸������r�h�i�m²¿��������������¿µ²¦¤¦°²²²²�������������������������|�w�v��������������óòö��������)�6�@�D�>�6�)�����'�4�4�6�4�'�'�������������@�F�L�S�Y�[�Y�L�@�=�3�3�3�6�@�@�@�@�@�@DoD{D�D�D�D�D�D�D�D�D�D�D�D{DnDYD\D]DgDo�'�*�*�3�'������'�'�'�'�'�'�'�'�'�'���������������������������{�y�x�v�y�}���#�/�<�H�U�a�d�n�o�q�o�a�H�<�/�#�!���#������������������������������������������������<�I�O�N�I�<�0����������������ؽݽ�������������ݽнĽ��������������C�O�\�h�o�h�\�Z�O�C�;�@�C�C�C�C�C�C�C�C���������������������������������������� O 0 K ( u ; > ! a � _ > $ ; # > P  " & d Z , q D   b } [ F R , M * P F O < 8 $ a T " k P * / M < k <  M  C  ;  �  S  �  �  h  V  �  �  w  �  �  �  �  n  �  �  �  �  }  +  �    E  E  3  }  �    �    �  �  {  �  9  �  �  �  !  z  L  u  ?  �  �  �  �  [  Q��h;D��<�C�;��
%   <T��=\)=o=<j<���=D��=q��=��`>��<�=aG�<��=��w=49X=�1='�=��=m�h=49X=��=T��=�9X=8Q�=49X=}�=>$�=��=e`B=q��=Y�=�t�=��=��P=��->�Q�=�%=���>49X=���=��T=��`=�/>��=�=�"�=�l�B!�aBe�A��zBB�B�B=�B vBۊB��BO�B�B�B�B��Bj�B!��B�B?�B�LB�B"�dB�B|AB#��B!��B!HhB�~B��B��B �B{�Bl�B��B�hBjIB ��B�B
��B�B�SB	�Bp8B�B��BͭB,�)BdBB�BD�B��B;HB!�/BCA���B?�B?�B��B ��B��B�8B�BB�B��B�B��B�JB"@	BαB:�B�zB#DB"��B��B�bB#�B!�~B!CsB6�B�B��B8�BGxB�AB�6BC�B@�B ��BDtBISB?2BGOB	:'B@]B�xB��BѼB,�DB@ABC�B?�B�hB��B<�@!�A�W�A�pA��A��WAæ�@eG%A�:�A��7AǛ3A�h�@��GA���C��6AI�LA���Ag�%A�9A8`A��@�0�@�� A�>�A@��A(�@���@�A3 �A��CAKΟC��@`�AalaA_ɥAh2A��AnG%Av��A��JA���A�Z�@�!�?�	C���?�r�A��A�=fA��:A�oA*.$Br�B��@#��A��`A�}|A���A���AÀ�@k:�A�o�A҅�A�{�A�y�@��A�h�C���AH�A�u�AgNA��A8�7A�qN@���@�-�A�d^A@�AA)	K@��e@��WA3MA��AKC��@�Aa�cA_H�Ah��A�$wAm�Av��A�{�A���AӀd@���?�C��X?�IAU�AĔCA��,A��A*�eB��B��            	                      !   '   Q   �            2      6      "         &      1            H   V      	            -        1         m   	         %   =                                          /      1   '                  )               '                     ;                  -      %   0                        %                                          '      !                                                         1                  '      %                           #         N��O�NOad`N�F�N 6�N���O:kO�O�JNS��P)O
W5O�aO��Nݎ�N���N456O�� Oq��O��N�dIOkN��N�z�O��Om�Ov@�M�u�N
�OY%O�YP:�!O���Np[ N��8N\��O���P	��N�^EPC�O��cM���NFqO5	aN|N�LOmvOʸ�Oԅ�OE]DN6��N3D  �  t  �  y  N  g  �    �  d  �  �     W  �  	    �  s  �  �  �     I  T  �  �  s  %  �  D  	  �      f  y    Y  �  �  �  �  L  �  �  z  �  	�  �  �  (�\)��`B��o�D���o<t�<���<�o<�j<u<��
=t�=aG�=ix�<�j=o<�/<�<�h=8Q�<��=+=�w=t�=49X=�w=D��=,1=,1=49X=��=y�#=<j=D��=D��=H�9=L��=aG�=Y�=]/>/�=e`B=��=��`=�C�=�\)=�t�=�t�=ě�=\=��=�G���������������������546:<@HUaahdaaWUH<55",/;CIJPMHE;80* �����


��������������� �������������������������������������������������������������� #/<DHUU_USH</##  ��������������������)(/Ha��������zaD<51) )+01.*)�������
������gedhnz�����������zng).45855)��������������������x~����������xxxxxxxx��|{����������������a_`et������������thanlnt���������������n����������������������������������������~���������������~~~~�������


	�������������������������������������������������������������������,//<>HKH</,,,,,,,,,,����������������������������

������������

�����������)..)������������"'*&��������

���������		
#/0/,$#
				�������������������� ��)47BXWRN5) cnw{{������������tgc����������������������������������������>88;BN[gt�����tg[NF>WR[[hphh][WWWWWWWWWWU[`hptxzvtihb[UUUUUU�������	

 �������%)6BMEB6-)%%%%%%%%%%������������������������������

��������),..)�����������! ����xuz����������������x

������������󺰺����������������������������������������������������������������������~�z�����T�a�m�v�z�}�y�m�a�T�;�/�$�"� �"�/�;�H�T�A�N�T�Z�f�Z�Z�N�A�6�5�1�5�9�A�A�A�A�A�A���������������������}������������������/�<�H�L�Q�U�X�U�H�<�/�-�%�$�/�/�/�/�/�/����-�1�:�B�:�-�&�!�����������������*�-�6�7�9�6�.�*�������������������������������������������n�z��z�r�zÇÈÇ�z�q�n�e�e�h�j�n�n�n�n���	���������������������������������������������������������r�q�l�r�v������)�5�B�U�]�]�Y�N�B�5�)����������� ��)FJFVFcFoFvF{FyFsFoFcFVFJF=F1F%FFFF0FJ����������������������������������������àìù��������ÿùìàÓÇÂÇÉÓØÝà�T�`�c�d�f�`�Y�T�G�E�G�I�T�T�T�T�T�T�T�Tù����������� ����������ùêãÛÜàìù��4�A�M�^�\�T�M�A�4�(��������������� �+�0�2�/�*�"��	���������������𻷻ûлٻѻлȻû������������������������4�@�M�Y�\�f�r�{�����r�f�Y�M�@�8�+�,�4�<�@�H�Q�O�H�<�;�/�*�#�!�#�$�/�1�<�<�<�<�M�Z�f�s�w�����~�s�m�f�Z�M�J�G�M�M�M�M�Ľݽ����� ������ݽĽ��������������ļ��&�'�+�)�������������������лܻ����ܻлû���������������������������
�������������
����
����������
�
�
�
�
�
�
�
�
�
�������ʾ׾߾�׾����������}�q�x�������E�E�E�E�E�E�E�E�E�E�E�E�E�E|EwExE�E�E�E����ֺ�ߺĺ������������v�e�L�K�O�d�������.�;�G�T�\�]�Y�T�P�G�.�"��	�����"�.��"�.�;�<�A�;�.�"������������T�`�m�m�r�m�h�a�`�_�T�G�E�C�G�O�T�T�T�T�/�;�H�P�R�H�D�;�6�0�/�.�/�/�/�/�/�/�/�/�y�������������������y�m�`�[�P�O�T�`�m�y�y�����ѿ�������꿸�������z�s�p�p�t�y²¿��������������¿¹²¦¥¦±²²²²�������������������������|�w�v��������������'�.�2�1�-��������������������'�4�4�6�4�'�'�������������@�F�L�S�Y�[�Y�L�@�=�3�3�3�6�@�@�@�@�@�@D�D�D�D�D�D�D�D�D�D�D�D{DoDjDiDnDoD{D�D��'�*�*�3�'������'�'�'�'�'�'�'�'�'�'�������������������������}�y�y�v�y�~�����#�/�<�H�U�a�d�n�o�q�o�a�H�<�/�#�!���#������������������������������������������������#�8�F�L�K�I�>�0��������������ݽ�������������ݽнĽ��������������C�O�\�h�o�h�\�Z�O�C�;�@�C�C�C�C�C�C�C�C���������������������������������������� O 0 N ( u ; B  C � R ,  1 # 6 P  "  d Y % T 8   b } [ 4 R , M * P F K < 8  a T # k H * / D < k <  M  C     �  S  �  1  E  Q  �  �  1  �  Z  �    n  n  �  �  �  r  �  �  =  E  �  3  }  �  L  �    �  �  {  �  �  �  �  �  !  z  x  u  '  �  �    �  [  Q  G
  G
  G
  G
  G
  G
  G
  G
  G
  G
  G
  G
  G
  G
  G
  G
  G
  G
  G
  G
  G
  G
  G
  G
  G
  G
  G
  G
  G
  G
  G
  G
  G
  G
  G
  G
  G
  G
  G
  G
  G
  G
  G
  G
  G
  G
  G
  G
  G
  G
  G
  G
  �  �  v  h  \  O  C  8  .  #    �  �  �  �  �  �  �  �  ~  t  s  s  p  m  h  a  Z  Q  F  9  ,      �  �  �  �  ~  _  z  �  �  �  s  ^  B  "  �  �  �  m  7    !  �  O  �  )   �  y  w  t  o  i  b  X  N  B  7  *         �  �  �  �  �  �  N  W  a  j  t  ~  �  �  �  �  �  �  �  �  �  �         .  g  f  e  d  d  _  Q  C  5  '      �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  s  U  2    �  �  �  @  �            �  �  �  �  j  >    �  �  �  Y  -  �  �    X  �  �  �  �  �  �  �  �  �  �  |  S    �  p  �  =  Z  m  d  Q  =       �  �  �  �  �  �  �  �  �  }  [  7    �  �  x  �  �  �  �  �  ~  m  X  A  +    �  �  �  �  Q  
  �   p  �  �  #  F  d  u  }  �  �  |  n  R  -  �  �  ,  �  '  �  y  �  :  n  �  �  �  �  �     �  �  v  !  �  ?  �    ?     �  �  �  �  )  O  W  M  8    �  }  �  P  �  �    	�  �  �    �  �  �  �  �  �  �  �  y  e  P  ;  )      �  �  �  x  >  �          �  �  �  �  �  �  �  Z    �  �  x  _  F           �  �  �  �  �  �  �  �  �  t  W  :  $     �   �   �  �  �  �  �  �  s  M     �  �  �  U    �  J  �  #  ^  =    s  h  \  O  ?  -      �  �  �  �  �  �  �  �  Y  )   �   �  h  �  �  �  �  �  �  �  �  h  I    �  �  C  �  �  �  R  �  �  x  n  b  V  J  =  /       �  �  �  �  x  v  u  u  u  v  �  �  �  �  �  �  �  �  �  `  0  �  �  P  �  H  �  �     �  �  �  �  �          �  �  �  �  m  6  �  �  T    �  n    %  /  :  C  H  ;  *    �  �  �  �  �  n  \  M  B    �    :  M  Q  T  I  9  "    �  �  �  d  7  �  �    �  �    �  �  �  �  �  �  �  s  [  D  ,    �  �  �  �  �  S  �   �  �  �  �  �  �  �  �  �  n  8  �  �  u    �    �    �  0  s  k  d  ]  V  N  G  =  2  &        �  �  �  �  �  �  �  %          	    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  m  J  $  �  �  �  �  h  3  �  �  �  �  {  i  U    �    7  A  C  6    �  �  N  �  G  �  
�  	�  �  �  y    �  �    	    �  �  �  �  �  �  w  &  �  ]  �  R  �  �  �  �  �  �  �  �  �    l  T  8    �  �  �  x  ;  �  �  2  �       �  �  �  �  �  �  �  {  X  +  �  �  �  =  �  �     �                   �  �  �  �  �  |  Z  C  8  <  @  D  f  W  G  7  (        �  �  �  �  �  �  �  �  �  v  g  Y  y  u  g  R  8    �  �  �  �  Z  )  �  �  v  0  �  �  -  �  �              �  �  �  {  S  "  �  �  _    �  f  %  Q  Y  T  <    �  �  �  y  M  "  �  �  �  S  �  p  �  �  �  �  �  �  �  �  �  o  c  _  R  E  @  B  :    �  �  H  �   �  '    ]  �  �  /  �  �  �  |  
  Z  f  3  �  �  �  x  �  F  �  �  �  �  �  ~  `  C  '  	  �  �  �  �  l  J  )    �  �  �  �  c  B     �  �  �  �  `  :    �  �  �  m  7    �  �  �    p  �    8  L  5    �  q    z  �    �  Z  	"  r  �  �  �  �  l  Y  F  4  "      �  �  �  �  �  �    '  b  �  �  �  �  �  �  y  b  F  )    �  �  �  �  \  9        J  z  f  I  )    �  �  ^    �  �  @  �  �  .  �  &  o  e  �  �  �  w  ^  G  0    �  �  �  n  9  �  �  z  *  �  �  4  �  	^  	�  	�  	�  	~  	]  	+  �  �  j  -  �  �  M  �  j  �    `  �  �  �  f  =    �  �  f  0  �  �  �  k  =    �  �  �  �  v  �  �  �  �  �  �  r  c  T  E  8  .  #    �  �  �  A  �  �  (  !      
    �  �  �  �  �  �  �  �  |  j  X  G  5  #