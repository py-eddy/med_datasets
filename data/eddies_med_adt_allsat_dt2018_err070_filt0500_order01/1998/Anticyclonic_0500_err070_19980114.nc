CDF       
      obs    4   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?pbM���   max       ?�5?|�i      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M� �   max       Pu:3      �  |   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       �,1   max       >�      �  L   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>Ǯz�H   max       @E�ffffg            effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?���P    max       @vc��Q�        (<   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @,         max       @M�           h  0\   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�L        max       @�x�          �  0�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �C�   max       >H�9      �  1�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A���   max       B+�q      �  2d   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�N   max       B+�]      �  34   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       ?�s   max       C� "      �  4   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       ?��   max       C���      �  4�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          v      �  5�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          9      �  6t   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          3      �  7D   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M� �   max       Pc�4      �  8   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��Ϫ͞�   max       ?�/��v�      �  8�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       �,1   max       >$�      �  9�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�\(�   max       @E�ffffg        :�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?���P    max       @vc33334        B�   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @/         max       @M�           h  J�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�L        max       @�&�          �  K,   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         D�   max         D�      �  K�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�u%F
�   max       ?�%F
�L0     @  L�   	   	                                 [      %   b            4      H   N   8   8   	   %   0         C   v   	      h                              '         /            DN�)�N�*N���N/��N��_N�M�N�L�N��vN���O$?N�~�NP��P5K*N�4�P3}Ppc\O$ŖN0e�N_��O�FKN��EPU<�Pu:3P&�O��cN]��O��O�O؋ORS<PYG�PQXN(`�N�<PZV�O�@
N��~M� �O|7O@m�N�dUN��N�nKN�OS^AO<��M�f�O��_N�3�NdB�Nh��O��,1�C������e`B�49X�#�
�t��t��o��o��o:�o;�o;�`B<#�
<49X<49X<49X<���<��
<�1<�9X<ě�<�`B<�`B<�`B<�h<��=t�=��=�w=�w=�w=#�
='�=49X=8Q�=P�`=}�=�C�=�\)=�\)=�\)=�\)=��P=��P=���=�v�=��`=��=�;d>�����������������������������������������~|z�������������~~~~"#*057;0#""""""""""!#/<@<;7/#!!!!!!!!��������������������%)5=BN[\[[TNB;5)%%%%%()/6BORYQOB6)%%%%%%olpt��������{toooooo������������9:<CHMUW]ZUHD<999999knt|�����tkkkkkkkkkk'6ht|}x��znfUB6!}yxwy���������}}}}}}JOT[t�����������h[QJ�����
#/BIOWYYQH<"�E><HSUaknruronfaURHE����������������������������������������tz}�������������||zt������������������������������������������'-26)����NKNUa�����������naUN��������������������##00<IE><10/)#######"/;BHC?2/&"	UTVOH/##/<@MU���!))--,)
��������������������������,35==5)���������;BNgmncY@)��ABNVWUQNMIEBAAAAAAAA��������������������������8EKOJB<5)�����������
#"
���746:<HPU[WURH<777777�������������������������������������������������������������������������������������������������������

����� #+/74//#!!'/6;HTZYTQIH;/+"!W\cchmmoz}������zmaW\bin{~}{nb\\\\\\\\\\��������������������-(%%)/2<@FHA</------�������������������������������������������������������������������������������������������������$�0�=�D�B�=�2�0�*�$�����$�$�$�$�$�$�a�n�zÇÓÕÕÓÇ�z�p�n�m�a�`�_�a�a�a�a�x�������������x�m�w�x�x�x�x�x�x�x�x�x�xD�D�EEEEED�D�D�D�D�D�D�D�D�D�D�D�D�Ϲܹ�����������ܹϹù��ùùϹϹ���������������������ŹŲŵŹ�����������ƺ����� �������������������L�Y�e�p�p�m�e�Y�L�@�<�<�@�D�L�L�L�L�L�L��*�2�6�:�<�6�1�.�*�����������E\EiEpEuEzEuEnEiE\EPEDELEPEUE\E\E\E\E\E\�������ûǻû������������������������������4�M�f������r�Y�;�4�'����ܻԻۻ���!�-�:�F�S�\�S�J�F�:�-�-�!��!�!�!�!�!�!�Z������������Z�<����ٽ�������4�Z�	�"�;�H�T�e�l�j�a�T�;����������������	������������������������������|�}������ûƻϻϻû����������ûûûûûûûûûü����ʼϼּ̼ʼ�������������������������������������������������ùæãé�������޼�'�(�-�2�2�'�$���������������������������g�N�5�(��
��5�A�Z�s���������#�0�I�Z�d�g�U�0���Ŀ��ļĞĚġ������(�A�T�g�t�������g�Z�N�(�����������������������������������������������������������r�o�f�e�f�r�~���������"�H�a�m�u�y�u�m�a�H�;�"��	� ���������l�`�G�;�6�4�8�G�T�`�m�y�������������y�l�������������������������y�w�q�u�y��������������'�)�.�-�'����������������ѿ��ٿʿ������m�`�L�D�@�T���������������2�<�O�[�g�T�B�6�����úõÿ����ĚĝĤĚčā�t�k�tāčėĚĚĚĚĚĚĚĚ�"�/�;�H�M�S�T�X�T�Q�H�D�;�2�/�%�"��"�"Ƨ������$�.�$�������Ƴ��h�^�_�f�rƎƧ�������ʾ׾����ݾʾ���������s�}�����������������������������������������H�P�R�K�H�;�;�9�;�D�H�H�H�H�H�H�H�H�H�HÓàìùþ��������ùìàÓÇÄÀÁÇÈÓ�y���������������������������y�s�s�m�q�y������������������������������������������������������������Ž����������������������
��������������������������)�5�9�B�H�B�5�)�)�!�)�)�)�)�)�)�)�)�)�)���
��#�+�0�5�;�8�#�������������������<�I�b�n�{ŇŋŔŖŇ�{�r�b�U�I�C�<�:�9�<�������
�����������������������������ʼּ���.�:�?�;�)�!�����ּ����������b�o�{ǈǔǗǔǎǈ�{�o�b�]�^�b�b�b�b�b�bD�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�EuE�E�E�E�E�E�E�E�E{EuEkEuEuEuEuEuEuEuEu�S�_�l�t�x�v�r�l�S�O�-�,�$�����!�:�S . M O J 3 ` j U  @ 3 7 O S f \  W * + : p A C < P k & 9 T N Q w I U X , S  > @ : P g . U r d < w B B  �  �  �  7  �  $     �  �  g  �  n  R  �  �  �  `  m  k  o  �  9    �  q  ~  <  n  8  �  �  �  y    A  8  �  ,  W  �    �  �  =  �  �  P  �  �  �  �  ŽC����ͼ�1�#�
;��
;o��o��o<�t�;�`B<�C�<D��=\<#�
=@�=�;d=+<�t�<���=�hs<�/=��=��=�1=�1=�P=��=��w=Y�=P�`=�
=>�w=@�=P�`>z�=��=�o=]/=�^5=�Q�=� �=�{=��T=��=�`B=�v�=��
>\)=>�>C�>H�9Bb�B'�B
��B%��B*SB1B��B�B��B�,B\7B�LB��B�lB+�BOB�B!͸B"��BK�B!�B�LB�eB��B�FB%�YA���B5�Br�B!�B��B��B"�B��B1?B$)BBB ;�B!�TB+�qB�BɄBW�B�@A��A���B(+SB��Bx�B�Bm}BS�BN�B?�B
F�B%GYB@�B��BՏB:B��B�fB?�B�VB�KB�B��B��B<|B"8lB"��B1�B!��B��B9ZB=AB��B&?�A�NB��B@�B ��B��B=�B@�B��B<B#�~B?BB 9LB"7�B+�]B"�B�KB@�B*�A���A�$B(7`B?�BK3B�,B@}B��A��B
�A� @�>�C�K?�sA�آ?e�?�!A�s�C���@�/O@��q@|_�A<�jA���AJt@�X}@��9Aϳ�@�V�A�rA�_A�hPA���@�A���Aii�AqD?Y�+Ao<�A�gA�#�A��NB�"AK6�A��XA�ePAˣ�AQ�A�o�A�wDA�.�A�b�A�A�kg@U�A��B,*C��"C� "@���A�*B
>�A�y�@��C�O�?��A�}?q+C?�TxA� C�ׅ@��_@�G1@���A>��A��AIk@���@��A��N@���A�~XA鰎A��%A�p�@�lA�.�Ah�Aq5?L�Am)�Aҕ�A�A��B�mAKZA�~�A�t�A˨tA��A�mgA���A���A��sA���A��@W�A��B,pC���C���@|E�   	   
                        	         \      &   c            4      I   N   9   9   
   %   0         C   v   	      i                              '         0            D                                       1      1   5                  9   9   )         #            2   1         1                                       #            #                                       !         /                     3   '                        #         #                                       #            N�)�N�*N<��N/��NV�?N�M�N�L�N��vN1�|O�mN�~�NP��O��N�4�O��P.ыN�$TN0e�N_��O=��N��EO�f�Pc�4P DO�W�N]��O�YO��N���ORS<O�"O��N(`�N�<O�ɡN�J�Nt�cM� �O|7O@m�N�_TN��N��sN�O2M�O(YM�f�O��_N�3�NP�rN2O�"  �  J  �  �    �  A  _    �  �  (  �  �  O  	  �  �  �  �  �  @  	  d  �  �  �  �  �  �  D  �  �  �  
�  �  f  �  V  h    �      		  �  a  k  �  
�  D  �,1�C����ͼe`B�t��#�
�t��t�;�`B%   ��o:�o=+;�`B<�j<�h<u<49X<���=t�<�1=D��<�`B<��=t�<�`B=+=��=#�
=��=�C�=��P=�w=#�
=���=e`B=H�9=P�`=}�=�C�=��=�\)=�hs=�\)=��w=���=���=�v�=��`=���=�`B>$�����������������������������������������}{���������"#*057;0#""""""""""" #/<=<:6/#""""""""��������������������%)5=BN[\[[TNB;5)%%%%%()/6BORYQOB6)%%%%%%qtw�������vtqqqqqqqq�������	�����9:<CHMUW]ZUHD<999999knt|�����tkkkkkkkkkk)+*+4B[hjjopmg`TOB0)}yxwy���������}}}}}}VSRSV[ht��������th[V����
#/>GPPOH<"
�CEHLUanpspnkaUJHCCCC����������������������������������������������������������������������������������������������������������%,./+�����PLPUan����������naUP��������������������##00<IE><10/)#######		"/;?EA=6/)"
	#/<HPQRH</*#�#))&�����������������������')-0.)��������)5BJPNG5)�ABNVWUQNMIEBAAAAAAAA������������������������)5>BCA;5)�����

������968<<HKUWURLH<999999�������������������������������������������������������������������������������������������������������

��������� #+/74//#!#)/9;HQTVWTNHE;.#"!^^addimqz|������zma^\bin{~}{nb\\\\\\\\\\��������������������-(%%)/2<@FHA</------�������������������������������������������������������������������������������������������������$�0�=�D�B�=�2�0�*�$�����$�$�$�$�$�$�a�n�zÅÇÐÇ�z�n�b�a�`�a�a�a�a�a�a�a�a�x�������������x�m�w�x�x�x�x�x�x�x�x�x�xD�D�EEEEEED�D�D�D�D�D�D�D�D�D�D�D�Ϲܹ�����������ܹϹù��ùùϹϹ���������������������ŹŲŵŹ�����������ƺ����� �������������������Y�c�e�h�e�b�Y�L�L�E�L�O�Y�Y�Y�Y�Y�Y�Y�Y��*�1�6�8�8�6�,�*�&�����������E\EiEpEuEzEuEnEiE\EPEDELEPEUE\E\E\E\E\E\�������ûǻû������������������������������'�4�M�`�o�k�f�Y�@�4�'����������!�-�:�F�S�\�S�J�F�:�-�-�!��!�!�!�!�!�!�A�M�Z�f�s������y�s�Z�A�4�(�����0�A��"�;�H�Y�a�c�a�T�;�����������������������������������������������������������ûƻϻϻû����������ûûûûûûûûûü����ʼϼּ̼ʼ���������������������������������������������������ùùïñù���ż�'�(�-�2�2�'�$�������������s�������������������������s�g�Z�S�[�g�s���#�0�I�X�a�d�U�0���������ĢĝĤ��������(�A�P�c�j�r�����s�g�N�(�������������������������������������������������������������r�o�f�e�f�r�~�������"�;�T�a�m�s�v�q�m�a�H�;�/���������"�m�y�����������y�m�`�G�;�9�8�;�?�J�T�`�m���������������������y�v�y�|��������������������'�)�.�-�'������������������������������y�m�`�X�S�R�\�`�m�������������%�.�5�7�5�)����������������ĚĝĤĚčā�t�k�tāčėĚĚĚĚĚĚĚĚ�"�/�;�H�M�S�T�X�T�Q�H�D�;�2�/�%�"��"�"Ƨ���������������������ƳƚƉ�ƂƎƧ���������ʾþ����������������������������������������������������������������H�P�R�K�H�;�;�9�;�D�H�H�H�H�H�H�H�H�H�HÓàìùþ��������ùìàÓÇÄÀÁÇÈÓ�y���������������������������y�s�s�m�q�y������������������������������������������������������������Ž����������������������������������������������������)�5�9�B�H�B�5�)�)�!�)�)�)�)�)�)�)�)�)�)�
��#�)�0�2�8�0�(�#��
��������������
�<�I�U�b�n�{ŅŇœŇ�{�q�b�U�I�D�=�;�;�<�������
�����������������������������ʼּ���.�:�?�;�)�!�����ּ����������b�o�{ǈǔǗǔǎǈ�{�o�b�]�^�b�b�b�b�b�bD�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�EuE�E�E�E�E�E�E�E�E}EuEtEuEuEuEuEuEuEuEu�S�_�l�r�w�u�q�k�S�:�-�&�!�����!�:�S . M L J * ` j U # F 3 7 D S B Z   W * ! : 1 @ B 1 P ` & ? T ; 8 w I @ , 1 S  > 8 : < g ) J r d < k 9 =  �  �  r  7  i  $     �  I  :  �  n  �  �  F  v    m  k  �  �  v  �  �    ~  �    �  �    �  y    0  �  �  ,  W  �  �  �  �  =  y  �  P  �  �  �  ]  �  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  �  |  p  d  Y  M  A  6  '      �  �  �  �  �  �  �  �  �  J  G  D  >  7  .  #          �  �  �  �  �  i  H  %    �  �  �  �  �  �  �  �  �  �  }  n  X  <  !     �   �   �     �  �  �  �  �  �  {  q  h  ^  T  K  A  8  /  #    	   �   �  �  �  	  	  �  �  �  �  �  �  z  `  C  &    �  �  �  p  P  �  �  �  �  �  �  �  �  }  e  L  1    �  �  �  �  �  �  �  A  5  )                	  �  �  �  �  �  �  n  >    _  Z  V  R  N  J  F  B  <  4  -  %    �  �  �  q  I  !   �  V  �  �  �  �  �            �  �  Z    �  r    �    �  �  �  �  �  �  �  z  p  f  \  S  K  F  @  <  9  ?  R  f  �  �  �  �  �  n  S  5    �  �  �  Y    �  �  5  �  �  >  (  *  ,  .  0  2  2  3  1  /  ,  H  v  v  r  n  i  d  _  Z  �  r  �  '  b  �  �  �  n  A        �  �  U  �  �  c  �  �  �  �  �  �  �  �  �  �  ~  t  i  _  V  N  G  @  8  1  *  �  �  �    3  C  L  M  E  6  "    �  �  �  8  �  S  �   �  �  �  	  	  	  �  �  �  �  r  9  	  �  �  =  �  #  &  R   �  y  �  �  �  �  �    k  Q  0    �  �  o  .  �  �  Q  -  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  y  �  �  �  �  �  �  �  �  �  �  �  �  q  b  R  A  /    �  �  #  l  �  �  �  �  �  �  �  �  �  V    �  8  �  �  �  ?  o  �  �  �  �  �  �    w  o  g  a  \  V  Q  J  C  4    �  �    1  ?  {  �    =  ?  :  )    �  �    �  �  Y  �  �  �  �  	   �  �  �  �  X    �  �  �  N  �  r  �  �  F  �  �  �  N  d  W  :    �  �  a  ,  �  �  �  J  �  �  3  �  V  �  �  �  �  �  �  �  �  �  �  \    �  �  ?  �  |  �  @  K     �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  c  �  �  �  �  �  �  �  �  ^    �  x    �  %  b  {  �  �  �  �  �  �  �  �  �  �  �  T    �  i  �  �  �  .  �  �  �  V  s  �  �  �  �  �  �  �  �  m  V  ;    �  �  �  Z  �  H  �  �  �  v  ]  F  /    �  �  �    U  (  �  �  �  �  R      �  �  �    
    *  ?  2  �  �  k    �  �    1  �  |  �  �  �  �  f  �  �  �  �  w  8  �  n  
�  
<  	�  �  �  �  �  �  �  �  �  n  U  6    �  �  �  �  q  O  -      �  �  �  �  �  �  m  Q  5    �  �  �  �  m  H    �  �  �  k  /  �  	�  
J  
  
�  
�  
�  
�  
�  
�  
�  
�  
?  	�  	�  	  r  �  k  �  h  �  �  �  �  �  �  �  �  �  �  �  �  V  "  �  �  u  3  �  C  4  F  V  `  f  _  F  *    �  �  D  �  �  �  �    �  9   �  �  �  �  �  �  �  �  �  �  �  �  �  �  �    n  ]  L  :  )  V  T  L  ?  $     �  �  Y    �  w  $  �  x    �  	  �    h  f  d  d  a  _  ]  V  L  >  !  �  �  q  +  �  s  �  �    �  �        �  �  �  �  �  �  �  �  �  �  �  �  �  �  	  �  �  �  �  x  f  P  4    �  �  �    R    �  �  )  �  `  	      �  �  �  �  �  �  b  >    �  �  �  \  )  �  �  �    	     �  �  �  �  �  �  �  �  �  �  �  s  O  +    �  �  �  	  		  	  �  �  �  �  x  O    �  �  X  �  3  m  �  �  �  �  �  �  �  �  �  �  j  M  *    �  �  �  P  �  �    �  2  a  X  O  F  =  4  +  #        �  �  �  �  �  �  �  �  �  k  N    �  �  1  �  \  �    �  �  �  x  /  �      d  �  �  �    b  F  (  
  �  �  �  {  U  /  	  �  �  �  �  �  �  
P  
�  
�  
�  
d  
:  
  	�  	�  	�  	>  �  f  �  r  �  :  �  �  L  �  "  B  ;  .      �  �  �  �  �  �  �  �  �  k  2  �  �  �  �  �  �  �  ;  �  �  4  
�  
l  
  	�  �          �  V