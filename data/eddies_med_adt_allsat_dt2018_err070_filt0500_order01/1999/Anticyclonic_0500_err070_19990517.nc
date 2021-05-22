CDF       
      obs    5   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?��"��`B      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       N>=   max       Pǫ�      �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       �49X   max       =��m      �  T   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>���
=q   max       @F��Q�     H   (   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�����̀   max       @v��Q�     H  (p   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @'         max       @Q            l  0�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�r        max       @�M�          �  1$   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ��o   max       >��^      �  1�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A���   max       B40�      �  2�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A���   max       B4;�      �  3�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       ?�bh   max       C��#      �  4t   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       ?���   max       C��H      �  5H   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          �      �  6   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          E      �  6�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          ;      �  7�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       N>=   max       P��      �  8�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��`A�7L   max       ?尉�'RU      �  9l   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       �49X   max       >%�T      �  :@   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>\(��   max       @F��Q�     H  ;   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�����̀   max       @v�z�G�     H  C\   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @"         max       @Q            l  K�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�r        max       @�4           �  L   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         Fq   max         Fq      �  L�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��Q�`   max       ?尉�'RU     �  M�         �   ,                  '         �   	      )   #   Y   $   >                                  	   
               (      	         %                  �         	   (   /N>=N�qPK/'O�t�N<��O/��Ns��N�OO_�O�O �QN1N�P��ENy�No�8Px}QOnFPǫ�O��KO��Of3�Oږ+O5��N�6�N��N���O�)O�N50�N N)WQOG�OE4N��O!��OA�P1�VN�gON��N��lOЃN$fO-J>N3r�O�`N���O�N�NX�N%��N���O.��O��׼49X�o%@  ;�`B;�`B<o<o<#�
<49X<49X<49X<T��<e`B<e`B<u<u<�C�<��
<��
<�1<�`B<�`B=+=+=+=C�=C�=��=�w=#�
=#�
=,1=0 �=49X=@�=H�9=L��=P�`=T��=T��=T��=Y�=]/=]/=e`B=y�#=��=�7L=���=�{=���=��m=��mipt}�����tiiiiiiiiii��������������������)0BN[g~�������tf[N6)���������������������������������������� ��	"/363+%"	)6;BDDB61)��������������������eipt������������tlge|{�����������������|� ����
#*.#
��fenyzz�znffffffffff)Oh���������hB)$)6BOPOIB>64-)$$$$$$ )/))	        'HUan�����naH# )8N[^gqtvtg[NB?5.) ����5B[ddYU\N)��	(/<HRVXXRH/
/<>Hanz���znaUHF@</������������������������
<GRWS<#
�����{y�����������������{������

����������T[`hntwyuth[TTTTTTTTmty������������tmmmm~{|~��������������""&/;HT^b`[THD>70/"%)06:BEOPOB62)%%%%%%�����

����������!#-/<AHJH</#!!!!!!!!����
#-.,(#
���
	!6@7896.)��������������������<ABFJO[hkpsrlh[OBB<<otz�����������|ztomo������/B@=0)���*+6<6,)%&)******	")1567525)(40155BIN[gig][VNB544������)����������������������������������������������������������

#










!)+5BM[gjkgd[[NB5)<:?BO[ad`[SOJB<<<<<<��������

��������������������������������������������������
��������������� 

 ����vnox��������������}v�������ûɻû��������������������������������������������������������)�B�a�p�s�o�[�6�������������������)ìù��������������ùìàÓÇÃÄÌÖÝì�׾���������߾ؾ׾Ӿ׾׾׾׾׾׾׾��H�T�a�m�q�y�z�����z�m�a�T�H�A�;�9�;�@�HÇÓÜÔ×ÓÊÇ�z�x�r�z�ÅÇÇÇÇÇÇ��#�/�5�:�6�1�/�#������������a�m���������������z�u�m�a�T�J�H�P�T�Z�a�4�M�Z�c�f�g�d�Z�J�A�4�2�(�������(�4�_�l�t�����������������x�d�_�S�R�S�^�]�_���������������������������������������޼@�Y�u�{�g�r������G� ���������'�@�ּڼ��ڼּӼʼ��������ʼϼּּּּּ־Z�\�a�f�d�Z�M�M�F�K�M�S�Z�Z�Z�Z�Z�Z�Z�Z�������	��!�����������������������������������������������������������������Ƨ�������%�I�A�6�������Ǝ�h�W�4�5�FƁƧ�������׾��̾�������s�l�i�f�a�f�s�~��F1F9F=FEFPFWFUFJF=FFE�E�E�E�E�E�FFF1ù�������������������������üòðù��&�-�8�0�#������������������������������@�8�A�D�A�4�(���� ���������(�2�,�0�+�(�$������������@�B�L�S�Y�[�Y�L�@�?�7�>�@�@�@�@�@�@�@�@�5�<�A�I�N�Z�e�Z�Z�N�K�A�6�5�-�.�5�5�5�5�������������������������������}�~�������#�0�6�0�(�+�+�$��
�������������
���#�����������������������������������������)�5�B�H�B�8�5�2�)�����������M�Z�f�g�s�u�v�u�s�f�Z�M�F�A�<�9�A�L�M�M�e�r�~���������������������~�q�e�\�^�Z�e��!�!�-�6�:�<�B�:�-�+�$�!��������:�F�I�S�_�f�k�f�_�S�F�:�-�(�%�#�-�.�:�:�������������������������������������������(�?�N�[�c�f�N�5����ٿѿҿݿ����?�3�'�'�"�����'�/�3�@�A�?�?�?�?�?�?�������ĿĿȿĿ��������������������������`�m�y�������������{�y�x�m�i�`�V�V�_�`�`�-�:�F�S�^�S�N�F�:�5�-�"�!���
��� �-���������Ľ˽߽۽ý����������y�s�o�m������!�#�$�#�!���
��
����������<�H�a�n�z��z�p�n�m�a�U�H�<�/�.�&�/�2�<�`�m�y�|�}�y�m�`�_�`�`�`�`�`�`�`�`�`�`�`�H�U�`�a�d�g�n�r�t�n�a�Y�U�H�H�G�G�E�H�H�ѿݿ������ݿѿɿĿ¿Ŀȿѿѿѿѿѿ�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�DwDsDvD{D��/�2�;�E�H�S�R�H�D�;�5�/�,�+�/�/�/�/�/�/�T�]�a�b�a�Y�T�H�G�D�H�I�T�T�T�T�T�T�T�T�
��#�0�1�<�B�B�<�0�*�#���
��
�
�
�
EuE�E�E�E�E�E�E�E�E�E�E�E�E�E}EuEoEjEiEu�����ʼּڼ������ּʼ��������������� b % 0 S F < g _ 6  8 + k F b X G J ' . B V [ [ n w G l f X d $ J x  D  S 2 > k 8 H A 0 o  ) � V q 1 ;    K     t  Y  j  �  �    �  �  0  H  �  �  �  6    �  "  �  �  2  �  �  x  �  a  �  �  U  �    �  �  X  �  �  O  B    I     J  �  B  �  �  �  �  ]  �  |  0��o<49X>+=L��<#�
=o<u<��
<�/=L��<�9X<�o>�+<�j<�9X=aG�=P�`=�;d=aG�=���=u=e`B=�P=�P=#�
=�P='�=L��=<j=0 �=H�9=P�`=u=D��=�C�=�C�=�E�=m�h=y�#=�%=�%=�E�=q��=��=y�#=��P=��->��^=�{=Ƨ�=�;d>&�y>-VB[�B��B	U�B!|3B40�A���Bs�B�>B
�cB 8xB$��B�B�SB�B��B�]Bk-Bo�BJSB;,Bv	Bg�B� B�B��B
R�B
�9A�hB��B<�Bu\B$��B��B*�BB%�B E\B��B=�BZ�B'B�B+�:B�CB��B�B��B��B�B2kB�vB�B��ByiB��B� B	I�B!��B4;�A���B�cB�hB
DB @'B$�IBhB�B0�BD�B:rBF;B�gB{KB<�B?�B|�B�B��B�B
�oB
�7A�kB��B>�B��B$��B�JB*�B@�B >�Bw�B@B;B��B�DB+��B�FB �B��B��B��B?�B��BBB@^B̹B1�@�w�A�<pAց�A��AUK�A�1Aɓ�A��xA�ֱA91@��A�~@Т�A ��A>�'A�WsA��DB��AJ.�C��#A�AӲ!A4��A6p�?�8�A��A���A�TA��A��A�]�A?*e@��@n2�@���A��$A��?�bhAt��Ak�6@v{dA �#A���A�S#AkGiA�"A|АC��LA���A�CjA�p�C���@�x6@�!�A�[A��A�}�AU2�A�|�AɃQA�A�v�A99@�|A�@��A sA>��A���A�l�B��AJ�C��HA�z�A��:A4�A4�?��}A�ahA���A�w5A��5A�85A��A>�@�t@lB@��,A��A���?���At�Ak�@pl�A#fA�}A���AkA�}eA|��C��:A��>A��A�}�C��A �:         �   ,                  (         �   
      )   #   Z   $   ?   !                              
   
               (      
         %                  �         
   )   /         /                     !         E         9      E   %         '                                             )               #                                          !                                       5      ;   !         #                                             '               !                                 N>=N��O�#KOi�pN<��N���Ns��N�(YO_�O$�N���N1N�O�+Ny�No�8PF4�O#�P��O�;CO�	LOU�O�J�N��N�6�N��N���O�)O�N50�N N)WQOG�O��N��O!��OA�P�gN�gNꔳN��N��lO��gN$fO-J>N3r�O�`N���O��NX�N%��N���O.��O���  �  �  �  K  T  w  N  p  �  �  �  �  
k  Q  �  N  k  F  '  
A  �  �  �    �    �    �  J  �  �  \  �  h  x  �  g    �  l  �  �  _  V  (  �  z  �  �  �  
Y  	��49X;o=e`B<49X;�`B<�o<o<49X<49X<���<e`B<T��=���<e`B<u<�1<���=\)<�j<�`B=\)<�h=C�=+=+=C�=C�=�w=�w=#�
=#�
=,1=<j=49X=@�=H�9=]/=P�`=Y�=T��=T��=q��=]/=]/=e`B=y�#=��>%�T=���=�{=���=��m=��mipt}�����tiiiiiiiiii��������������������ABDKN[gt������tg[LFA����������������������������������������	"/-%"	)6;BDDB61)��������������������eipt������������tlge��������������������	
#'*$#
								fenyzz�znffffffffffaghqt����������tjha$)6BOPOIB>64-)$$$$$$ )/))	        1HUan����naU<#5BINU[`glngg[NEB<655���5NWZRMMOA5)��!+/<EPUWUNH</#D<=CHanz|~���ynaUPID������������������������
<FQVR<#
�������������������������������

����������T[`hntwyuth[TTTTTTTTmty������������tmmmm~{|~��������������-/;HST]aaa_ZTHE?;2/-%)06:BEOPOB62)%%%%%%�����

����������!#-/<AHJH</#!!!!!!!!����
#-.,(#
���)156763)��������������������<ABFJO[hkpsrlh[OBB<<otz�����������|ztomo������*4==8)����*+6<6,)%&)******	)/5565/)		40155BIN[gig][VNB544������)����������������������������������������������������������

#










!)+5BM[gjkgd[[NB5)<:?BO[ad`[SOJB<<<<<<�������

 ����������������������������������������������������
��������������� 

 ����vnox��������������}v�������ûɻû�������������������������������������������������������������)�6�B�O�S�\�Z�O�6�)�������������)ù����������������ùìÓÇÆÏÙàìïù�׾���������߾ؾ׾Ӿ׾׾׾׾׾׾׾��T�a�g�m�q�u�m�a�T�N�H�C�H�J�T�T�T�T�T�TÇÓÜÔ×ÓÊÇ�z�x�r�z�ÅÇÇÇÇÇÇ�#�/�4�9�5�0�/�#�!������#�#�#�#�#�#�a�m���������������z�u�m�a�T�J�H�P�T�Z�a�4�A�E�M�V�Y�V�M�A�<�4�(������(�,�4�l�x�����������x�l�k�e�j�l�l�l�l�l�l�l�l���������������������������������������޼4�@�@�M�Q�Y�Z�Y�P�M�@�4�1�'�%�#�%�'�)�4�ּڼ��ڼּӼʼ��������ʼϼּּּּּ־Z�\�a�f�d�Z�M�M�F�K�M�S�Z�Z�Z�Z�Z�Z�Z�Z�������	��������������������������������������������������������������������Ƨ�����'�0�2�������ƳƎ�u�\�O�I�O�\ƁƧ�������ʾ׾޾۾ʾ�������s�o�n�l�l�s���FF$F1F=FBFMFSFNFJF=F$FFE�E�E�E�E�E�F������������������������������÷ù������#�,�5�.�"���������������������������(�4�:�4�0�2�(���
����������(�2�,�0�+�(�$������������@�B�L�S�Y�[�Y�L�@�?�7�>�@�@�@�@�@�@�@�@�5�<�A�I�N�Z�e�Z�Z�N�K�A�6�5�-�.�5�5�5�5�������������������������������}�~�������#�$�)�*�#�"���
����������������"�#�����������������������������������������)�5�B�H�B�8�5�2�)�����������M�Z�f�g�s�u�v�u�s�f�Z�M�F�A�<�9�A�L�M�M�e�r�~�������������������~�y�r�a�e�e�b�e��!�!�-�6�:�<�B�:�-�+�$�!��������:�F�I�S�_�f�k�f�_�S�F�:�-�(�%�#�-�.�:�:�������������������������������������������(�9�L�W�^�`�N�A�5����޿ܿ�����?�3�'�'�"�����'�/�3�@�A�?�?�?�?�?�?���������ĿĿĿ��������������������������`�m�y�������������{�y�x�m�i�`�V�V�_�`�`�-�:�F�S�^�S�N�F�:�5�-�"�!���
��� �-�������׽ս˽Ľ������������y�s�q�y�������!�#�$�#�!���
��
����������<�H�a�n�z��z�p�n�m�a�U�H�<�/�.�&�/�2�<�`�m�y�|�}�y�m�`�_�`�`�`�`�`�`�`�`�`�`�`�H�U�`�a�d�g�n�r�t�n�a�Y�U�H�H�G�G�E�H�H�ѿݿ������ݿѿɿĿ¿Ŀȿѿѿѿѿѿ�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D��/�2�;�E�H�S�R�H�D�;�5�/�,�+�/�/�/�/�/�/�T�]�a�b�a�Y�T�H�G�D�H�I�T�T�T�T�T�T�T�T�
��#�0�1�<�B�B�<�0�*�#���
��
�
�
�
EuE�E�E�E�E�E�E�E�E�E�E�E�E�E}EuEoEjEiEu�����ʼּڼ������ּʼ��������������� b # ' P F ; g [ 6  / + : F b V : M   & ? W ( [ n w G D f X d $ 3 x  D  S & > k = H A 0 o   � V q 1 ;    K  �  �  �  j  �  �  �  �  V  �  H  9  �  �  �  ]  �  �  T  V      �  x  �  a  E  �  U  �    7  �  X  �  �  O      I  z  J  �  B  �  �  @  �  ]  �  |  0  Fq  Fq  Fq  Fq  Fq  Fq  Fq  Fq  Fq  Fq  Fq  Fq  Fq  Fq  Fq  Fq  Fq  Fq  Fq  Fq  Fq  Fq  Fq  Fq  Fq  Fq  Fq  Fq  Fq  Fq  Fq  Fq  Fq  Fq  Fq  Fq  Fq  Fq  Fq  Fq  Fq  Fq  Fq  Fq  Fq  Fq  Fq  Fq  Fq  Fq  Fq  Fq  Fq  �  �        %  ,  0  1  /  )      	  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  i  I     �  �  �  y  T  0  
  
�  x     �  $  v  �  �  �  �  Y  �  \  �  
�  	[  �  �  �  !  A  J  C  :  $        �  �  �  |  ?  �  m  �  �  {  [  T  Q  M  J  F  B  ?  7  -  #         �   �   �   �   �   �   �  �    ?  Y  h  r  v  v  o  `  E    �  �  R  �  �  .  �  0  N  A  4  &      �  �  �  �  k  H     �  �  �  ~  Q  "   �  H  ]  p  o  m  f  ^  S  H  >  1  "    �  �  �  �  f  '  �  �  �  �  �  �  p  ]  L  E  9  +         �  �  �  ~    �  �  �  '  K  j  �  �  �  �  �  �  p  [  <    �  {  �  +  l  �  �  �  �  �  �  �  �  �  �  �  �  �  �  }  h  R  <    �  �  �  �  �  �  �  |  p  d  X  L  ?  3  '        �  �  �    _  �  	  	f  	T  	>  	=  	G  	�  
  
&  
b  
j  
J  	�  	Y  �  !  _  Q  D  8  *      �  �  �  �  �  �  u  b  P  C  F  O  �  �  �  �  y  n  x  �  �  x  n  _  N  >  -      �  �  �  �  �  '  >  J  L  >      �  �  �  �  �  �  �  l  ]    �  E  �  �    9  P  `  j  a  O  4    �  �  f    �  @  �  H  �  u  �    5  E  ;  (    �  �  �  p  V  V  J    �  D  a  3  +    "  %  !         �  �  �  �  �  �  Q    �  ]  �  Y  �  
  
/  
A  
6  
&  
  	�  	�  	  	!  �  Q  �  z  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  g  +  �  �  X    �  P  �  <  �  �  �  �  �  N  ;    �  �  7  X  j  N    �  u    �  J  �  �  �  �  �  �  �  �  �  �  �  �  |  k  Z  H  ?  8  1  *  #             �  �  �  �  �  �  �  �  �  �  �  g  B     �  �  �  �  �  �  �  �  �  �  v  j  ]  K  :  !     �  �  �        �  �  �  �  �  �  �  �  �  �  �  �  �  |  t  l  e  ]  �  �  �  q  ^  K  7  #    �  �  �  �  �  �  �  �  �  �    �  �        �  �  �  �  �  o  P  0    �  �  �  �  ]  6  �  �  �  t  ]  F  6  *      �  �  �  �  �  x  ]  @  "    J  F  B  >  :  5  1  .  +  )  &  #  !           �   �   �  �  �  �  �  �  �  �  �  �  �  �  �  �  |  r  h  _  U  M  D  �  �    u  i  \  O  @  1      �  �  �  �  d  :     �   �     )  4  Q  Y  N  ;  $    �  �  �  �  p  A    �  �    =  �  �  �  �  �  �  �  �  �  �  �  w  m  c  Z  B  &  
   �   �  h  ^  T  I  =  .      �  �  �  i  0  �  �  ;  �  �  (  �  x  u  j  X  =    �  �  �  z  T  &  �  �  m  #  �  M  �    x  �  �  �  �  �  l  K    �  �  e  "  �  �  T  �  �  8  �  g  c  _  V  >  '        �  �    y  �  n    �  /  �  n  
          �  �  �  �  �  �  �  m  R  5    �  �  �  o  �  �  �  �  �  �  �  }  k  X  C  ,    �  �  �  Y    �  �  l  g  e  j  k  i  ]  O  C  6  +        �  �  �  }  :  �  �  �  �  �  �  �  �  r  I    �  y    �  7  �  �  �  @  �  �  �  �  �  �  x  g  W  D  /      �  �  �  �  �  �  o  Y  _  O  @  2  (       �  �  �  �  j  A    �  �  z  K    �  V  L  B  8  /  &          �  �  �  �  �  �  f  >    �  (    �  �  �  v  ;  �  �  �  �  �  �  �  �  �  >  �  �  T  �  �  �  �  �  �  �  r  `  J  2    �  �  �  }  P  "    �  �  �  �  m    �  �  7  i  z  Y  �  [  �  h  �  `  �  2  	�  �  �  �  �  �  �  �  �  �  �  ~  z  v  r  n  j  f  b  ^  Z  �  e  I  2    �  �  �  �  �  k  R  P  N  M  J  G  C  ?  ;  �  �  �  q  U  6    �  �  �  o  ?    �  �  y    �  F   �  
Y  
R  
E  
1  
  	�  	�  	�  	4  �  �  8  �  r  �  l  �  �  �  �  	�  	m  	8  �  �  b  ]  C  +      �  �  +  �  O  �  _  :  2