CDF       
      obs    2   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?�bM��      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M�K�   max       P��<      �  t   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ���
   max       =�      �  <   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>Ǯz�H   max       @E�p��
>     �      effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��z�G�    max       @vs\(�     �  '�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @*         max       @N@           d  /�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�}        max       @�Y�          �  0   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ��o   max       >��-      �  0�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A��   max       B,�F      �  1�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�uP   max       B,�3      �  2`   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       ?M s   max       C���      �  3(   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       ?g�k   max       C���      �  3�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          �      �  4�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          =      �  5�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          ;      �  6H   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M�K�   max       P�r�      �  7   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?���?   max       ?���"��a      �  7�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ��o   max       >'�      �  8�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�
=p��   max       @E�(�\     �  9h   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?��Q�     max       @vs\(�     �  A8   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @#         max       @K�           d  I   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�}        max       @���          �  Il   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         F�   max         F�      �  J4   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�	� �   max       ?��	k��~     �  J�               2                                 c      F   )   /      $   @   :         1                   G      %         2   �       \   '   	   	                  N[8]O�nLOP��OK��Pl#�O��LO��N��FNy'TN���N��NAN�5Pj�OCAP��<O
�P��O�UO�.�N<`3O��	O��}P���N���N��yO��`Nq��ND}fO⇉NسYO�'P9OWOi��O�n�ND5O��P:GkO�n�OzbO�D`N�@N8�WO�lNh��N�t@N�6EN1��M�K����
��o��o;�o;�`B;�`B<o<49X<D��<D��<T��<u<�o<�o<�t�<���<�1<ě�<ě�<���<�/<�h=o=\)=\)=\)=t�=t�=t�=�w='�='�=0 �=@�=H�9=]/=e`B=ix�=m�h=m�h=y�#=}�=�%=�o=�7L=�7L=��T=��T=��=���������������������<BN[gkt}�~{t[NICB<5<"/4;;7553/"	#/<>AA?:2/#JI`t�����������ti^XJ�������������������5??Ibu{�����{bUI<605����
#%+,(#

�����)1/*)&��������������������st|���������vtssssss��������������������hiknwz{zvz|ztnhhhhhh��������������������Z[\bhtt���������th[Z������5NYWOB5����#$*/<HJSRHD<1/#vx�����������������v����)BLHA;7-)���������������������������������������������������#$
��������
#/;HTTQH<#
�����)5FNRTNE������������������������������������������������)5<KLG5)��	
   
						����������������������� ��������� �����)6BOQS][OB6/)$.)���������$��������������
 %(#
���z{�����������������z���������������������


���������������3-.9B[t���������gNB3C@?HUa���������xaUHC���������	
	�����3127;>HTam~����zaH;3����� �������������������������������� )45865.%	'()*5BCKDB50))''''''����������������������������������������ncbknrz~|znnnnnnnnnn# #'/21/###########¿¿��������¿²«¦¢¦²¼¿¿¿¿¿¿����� ��������������������������T�a�m�u�z�����z�m�a�T�H�/�"��*�;�H�P�T�����������������������������������������H�T�a�e�]�T�?�9�4�/�	��������������;�H�"�/�;�I�T�]�S�D�A�;�/�"��	��������"�������������������r�f�]�\�f�e�h�c�f�p���(�5�@�A�E�I�C�A�5�(������&�(�(�(�(�Z�f�g�g�f�a�Z�M�A�>�A�D�M�Q�Z�Z�Z�Z�Z�Z������������������z�x�u�z�|��������������������������������������������������������� ������ݿܿڿݿ�����������������������������������������������޾ʾ�	�"�*�8�7�(��	����׾˾����������ʺ3�9�@�L�Y�e�f�l�n�e�Y�L�K�@�3�3�+�'�/�3����I�e�k�o�l�`�M�#�
������įĸ��������������������	�������������������������(�7�O�]�j�k�]�N�A�5������������Y�e�~�������úκ������~�r�e�Y�Q�L�G�M�Y����'�;�K�I�@�2�'�������Ϲ����ù۹��5�A�G�M�J�A�6�5�.�0�5�5�5�5�5�5�5�5�5�5�B�[�g�t�~�}�t�i�c�[�B�)��� ����
�)�B�m�y���������������v�m�`�T�K�E�?�C�M�T�m������!�(�%������Ƴƚƀ�u�L�M�X�hƎ��¿��������������¿²¦¢¦²»¿¿¿¿àìïùý��ùöìàÓËÐÓ×Øàààà������������ ���������������������������T�a�m�s�z�������z�m�a�T�T�S�T�T�T�T�T�T�.�;�G�T�^�T�K�G�;�.�*�'�.�.�.�.�.�.�.�.��#�<�H�Z�a�p�p�a�U�H�<�/�%��	���
��S�_�h�l�x�z�x�m�l�_�S�F�:�:�3�:�F�H�S�S�-�4�0�-�$�#�!����������
��!�*�+�-��������ʼۼ޼�߼̼��������v�s�v�p�q������ʾ׾�����ƾ�������������������ìù��������������������������ùñìçì������/�;�H�a�e�g�a�X�T�H�A�"��������EuE�E�E�E�E�E�E�EuEuEuEuEuEuEuEuEuEuEuEu�G�S�`���������������������y�`�T�K�C�B�G�����6�?�O�S�O�G�6������������������(�A�Z�e�e�c�R�A�4�(�������������(D�D�D�D�D�D�D�D�D�D�D�D�D�D�D{DoDsD{D�D�ĦĳĿ��������������������ĿķīĨĢĢĦ�<�@�G�H�Q�H�<�/�#�#��#�/�6�<�<�<�<�<�<�B�O�[�_�h�i�h�[�O�L�B�>�B�B�B�B�B�B�B�B�����������������������������������������m�y�|�y�p�m�m�`�T�R�M�T�`�a�m�m�m�m�m�m�s���������������������s�n�s�s�s�s�s�s�s�N�Z�a�\�Z�S�N�A�5�4�1�5�A�E�N�N�N�N�N�N�ܻ������������ܻܻܻܻܻܻܻܻܻ�E�E�E�FE�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E� +  W 5 C P C > B Y A S o / U  E + O b Q C 2 8 H 4 9 ` Y 2 A ^ 9 g J y C +  L < @ . `  Y d M $ E  `    �  �  �    �    �  �  �  y  ?  �  [  '  >  �  4  i  m  L  �  �  �  �  �  �  f    �  z  �  L  �  �  >  �    N  �  }  �  �  S  �  �  �  O  ��o<T��<�C�<�1=e`B<���<���<�/<�t�<�C�<���<�9X<���=<j=�w==49X=�v�=�o=��<��=�o=\=�j=T��=8Q�=�1=�w=�w=�hs=Y�=�o=�x�=���=� �=�\)=�7L=�"�>��-=�E�>��=��=�t�=�t�=��
=�t�=�1=�{=�S�=��#B L�B�	A��B�B$sB��B'��BBBRTBr�BeB+��B�<B ��Ba�BZ�B��BEB��B�vB��B�pB׺B��B�B!�B*�B�Bb�B]�B�B3�B�B��BN^B�UB�B,�FB	8wB�8B�A��sB�OBK7B>	BN�BXBAOB�)B�B B@B�BA�uPB0[B
��B>{B'@)B3�BG�B��B;QB+��B�VB S�BR�BJB�B@B�WBBB��B�nB�B;lB�EB":BE�B5lBD�B��B�YBA�B�-B>�BBEB@�B��B,�3B	?�B��B��A��B/MB=kB؟Bz*B ��B?<BĹB"oA��A�U�A�A���A��A���@�CyA�LA>)�A���Ar�KA\�A��AXZ?���A�X�A�˷A�1�@
�}?M sA�8<A��~Ak�Bv�A�"�A���A��+A�`�Ac�3A��@�'-@]��@�AO�?AϘ�A���C��|A��A��A9��C���A�4�A��A���As��Ai��A�u�A�$�@�C���A�?3A���A�oRA���A���A�D�@���A�wA>�A���As A~�CA�FoAX�?�"�A�TA�R:A�\Z@�?g�kA�A���Ak{B:�A�x A��>A��A�نAc�A���@�i�@[�@��AQ<%A�yA��C��GA��A�1,A9��C��)A�}�AyA��At��Ah�cA��1A�1Y@��C���               2                                 c      F   )   0      $   @   :         1         !         H      &         3   �       \   (   
   	                                 5   #   '                     '      =      '   %   +      %      ;                  #         )         !      %   +   #                                             -   #   '                           5      !      #      %      ;                  #                        %      !                              N�6O�nLN��O'x�PE�`O��LO��N��FNy'TNHs�N��NV:N�5O�>ON�4�P�EFN�$�O�k~O��8O��N<`3O��	O�]�P�r�N���N��yO�~%Nq��ND}fO⇉NسYN��OD�O<��O@�OH,�ND5O�X�O��,O�Y9N�:�OWN�@N8�WO�lNh��N�t@N�6EN1��M�K�  �  r  z    B  p  �  l  �  �  6  �  1  �  �  .  -  �  �  E  *  �  
m  *  �  �  �  �  �  9  �  /  �  �  -  �  �  �  �  �  �  �  �  �  R    9  y  -  ػ�o��o;��
;ě�<D��;�`B<o<49X<D��<T��<T��<�o<�o<���<�1=C�<�`B=#�
=\)=+<�/<�h=#�
=t�=�P=\)=8Q�=t�=t�=�w='�=0 �=�1=H�9=ix�=ix�=e`B=m�h>'�=u=�v�=�\)=�%=�o=�7L=�7L=��T=��T=��=���������������������<BN[gkt}�~{t[NICB<5<"/4520/"$/8;<?@=<80/#`\^[^gt����������tl`�������������������5??Ibu{�����{bUI<605����
#%+,(#

�����)1/*)&��������������������st|���������vtssssss��������������������hiknwz{zvz|ztnhhhhhh��������������������^_dhjtz��������~th^^�������)BKRPH5���%$*/2<DHOLH=</%%%%%%�����������������������)5::60&����������������������������������������������������#$
������� 
#/CHMOLH<#
����)5ENSND�������������������������������������������������)7?DE?5)��	
   
						����������������������� ��������� �����&)06BOOR\[YOGB<60,)&�����������������������
##��������������������������������������������


����������������@AFQ[gt�����tg[NJB@EABHUan�������{naUHE����������������B>?CHTamtz���{zmaTHB����� �������������������������������� )45865.%	'()*5BCKDB50))''''''����������������������������������������ncbknrz~|znnnnnnnnnn# #'/21/###########¦²¿��������¿²°¦¥¦¦¦¦¦¦¦¦����� ��������������������������T�a�m�n�v�u�m�a�T�H�@�?�H�I�T�T�T�T�T�T������������������������������������������/�;�H�T�]�a�X�:�-�"��	��������������"�/�;�I�T�]�S�D�A�;�/�"��	��������"�������������������r�f�]�\�f�e�h�c�f�p���(�5�@�A�E�I�C�A�5�(������&�(�(�(�(�Z�f�g�g�f�a�Z�M�A�>�A�D�M�Q�Z�Z�Z�Z�Z�Z�������������������z�z�w�z�������������������������������������������������������������������߿ݿܿݿ�����������������������������������������������޾���	��"�'�)�&���	����׾ϾǾľʾ׾�3�@�L�T�Y�d�e�h�i�e�Y�O�L�@�7�3�-�*�3�3������0�I�X�e�c�Y�I�<�#�
����ļļ�������������������������������������������(�5�A�N�S�^�a�Z�N�A�5�(�����������r�~�������������������~�r�e�\�X�Z�e�k�r�ùܹ���5�@�F�E�@�3�'������Ϲ��������5�A�G�M�J�A�6�5�.�0�5�5�5�5�5�5�5�5�5�5�B�[�g�t�~�}�t�i�c�[�B�)��� ����
�)�B�`�m�y�������������y�q�m�`�T�P�H�C�H�R�`��������$�#������ƚƁ�u�N�O�Y�hƎƳ��¿����������¿²¦ ¥¦²½¿¿¿¿¿¿àìïùý��ùöìàÓËÐÓ×Øàààà�����������������������������������������T�a�m�s�z�������z�m�a�T�T�S�T�T�T�T�T�T�.�;�G�T�^�T�K�G�;�.�*�'�.�.�.�.�.�.�.�.��#�<�H�Z�a�p�p�a�U�H�<�/�%��	���
��S�_�h�l�x�z�x�m�l�_�S�F�:�:�3�:�F�H�S�S���!�*�"�!�!�������������������������¼ż��������������������������������ʾ׾�����׾þ�����������������ú��������������������������ýùõôøú�	��"�/�;�H�T�a�d�a�[�T�H�;�"�����	EuE�E�E�E�E�E�E�EuEuEuEuEuEuEuEuEuEuEuEu�G�S�`���������������������y�`�U�K�D�C�G���)�6�=�B�@�6�)������������������(�M�Z�b�c�a�\�I�A�4�(����������!�(D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�ĳĿ��������������������ĿĳĲĮĭīĭĳ�<�@�G�H�Q�H�<�/�#�#��#�/�6�<�<�<�<�<�<�B�O�[�_�h�i�h�[�O�L�B�>�B�B�B�B�B�B�B�B�����������������������������������������m�y�|�y�p�m�m�`�T�R�M�T�`�a�m�m�m�m�m�m�s���������������������s�n�s�s�s�s�s�s�s�N�Z�a�\�Z�S�N�A�5�4�1�5�A�E�N�N�N�N�N�N�ܻ������������ܻܻܻܻܻܻܻܻܻ�E�E�E�FE�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E� 6  - 1 : P C > B ^ A V o  S  < 0 8 b Q C 0 4 E 4 - ` Y 2 A 0  Y 8 g C +  V ; 4 . `  Y d M $ E  8    �  i  �    �    �  �  �  O  ?  a    _  �  �  %  0  m  L  R  �  �  �  U  �  f    �    -  �  ^  �  >  �  ;  �    �  �  �  S  �  �  �  O    F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  p  t  x  {    �  }  y  u  q  b  J  1       �  �  �    _  r  f  Y  M  G  C  >  8  2  ,  !    �  �  �  s  D    �  �  Z  U  M  6  <  W  q  x  n  Z  A  $    �  �  ?  �  _  �  5  �          �  �  �  �  �  �  �  �  m  B    �  �  �  5  )  9  B  >  7  7  -      �  �  p  .    �  �  �  ]  �  c  p  g  ]  Q  F  ;  .         �  �  �  �  �  �  �  �  q  P  �  �  �  �  d  E  #    �  
    
  �  �  �  e  '   �   �   M  l  U  =  "    �  �  �  �  n  J  &    �  �  x  d  o  v  ~  �  �  �  }  s  g  [  N  B  5  %    �  �  �  �  �  [  7    �  �  �  �  �  ~  v  o  d  S  C  3  !    �  �  �  �  �  �  6  .  &          �  �  �  �  �  �  �  a  C  "   �   �   �  i  q  z  �  �  �  u  \  B     �  �  �  �  u  R  0     �   �  1  1  1  1  1  2  2  -  %          �  �  �  �  �  �  �  }  �  �  �  �  �  �  �  �  �  �  �  _  ,  �  �  c    �    �  �  �  �  �  �  �    b  =    �  �  �  q  G  "  �  V  r  �    %  .  %  
  �  �  `  	  �  r  ^  B    �  �  �    �  �  �  
    $  )  ,  &    �  �  �  �  B  �  �  \  �  x  B  �  �  �  �  �  �  �  �  v  A    �  �  ,  �    �  �  �  �  /  K  L  I  �  �  �  �  z  a  <    �  �  P    �  x  �  =  �  $  =  E  B  7  !  �  �  �  i  3    �    �  %  \  �  q  *  %             �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  c  B    �  �  �  J  �  �  M  �  �  J  �   �  
C  
]  
k  
m  
e  
S  
2  
  	�  	�  	<  �  v  �  Q  �  �        )  %      �  �  |  H     )    �  �  l    �  e  �  L  �  �  �  �  �  w  c  H  )  
  �  �  �  �  a  =    �  J  �  l  �  �  �  �  �  �  �  z  b  H  -    �  �  �  �  �  x  X  5  �  �  �  �  �  �  �  �  ~  H    �  X  �  V  �  K  �  �  �  �  �  �  �  �  w  m  g  d  `  ]  Z  V  V  [  `  e  i  n  s  �  �  �  �  �  �  �  �  �  �  �  �    {  x  t  q  m  j  g  9  1  "    �  �  n  3  �  �  |  >  �  �  ^    �  p  �  �  �  �  �  �  |  c  B    �  �  �  p  =    �  u    �  I   �  *  o  *    �  �  �  ~  O    �  �  v  D    �     �  �  C  U  f  �  �    w  �  �  �  �  �  �  �  Y  �  l  �    ^  �  �  �  �  �  �  �  �  m  A    �  n  "  �  �  :  �  �  9   �         +  -  +  !    �  �  c    �  #  �  !  �    6  .  �  �  �  �  �  �  �  q  Q  0    �  �  �  ]    �  �  �  8  �  �  �  �  }  W  -    �  �  w  G    �  �  m  ,  �  �  f  �  �  �  �  m  G  2    �  �  �  y  e  V  F  (  �  C  Q  8  �  f  a  @  �  �  	  k  �    >  �  G  g    m  �  �  �  T  }  �  �  �  {  l  Z  G  2    �  �  �  �  y  <  �  �  $  �  �    T  �  �  �  �  �  �  �  y  4  �    :  �  
V  �  �  h  U  �  �  �  �  �  �  �  �  q  8  �  �  Y  �  �  �  +  `  �  �  �  �  �  �  �  �  |  }  ~  �    u  d  F  &    �  �  '  �  �  �  �  �  s  Z  A  )      @  `  U  J  @  6  -  %    R  D  4  !    �  �  �  �  i  D    �  �  �  s  O  $  �  e      �  �  �  �  �  �  �  �  �  �  �  p  W  >    �  �  �  9  /  &      	  �  �  �  �  �  �  �  �  �  �  �  �  y  m  y  j  Z  K  ;  +      �  �  �  �  �  �  g  X  L  @  5  )  -    �  �  �  �  �  {  \  :    �  �  �  J    �  m  )  �  �  �  �  �  �  �  �  �  �  �  �  {  p  f  \  S  I  R  b  q