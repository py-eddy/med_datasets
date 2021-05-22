CDF       
      obs    3   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?�^5?|�      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M�k�   max       P�ݔ      �  x   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ����   max       =���      �  D   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�\(�   max       @F�����     �      effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?��
=p�    max       @vg33334     �  (   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @*         max       @O�           h  0    effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @ʐ        max       @�+@          �  0h   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �o   max       >^5?      �  14   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A���   max       B.��      �  2    latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�x�   max       B.�v      �  2�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       ?3�   max       C��      �  3�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       ?�   max       C��      �  4d   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          �      �  50   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          E      �  5�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          /      �  6�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M�k�   max       P'S      �  7�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�U2a|�   max       ?�iDg8~      �  8`   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       �T��   max       =�/      �  9,   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�33333   max       @F�����     �  9�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?��Q�    max       @vf�Q�     �  A�   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @*         max       @O�           h  I�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @ʐ        max       @          �  JP   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         Bz   max         Bz      �  K   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�bM���   max       ?�hr� Ĝ     �  K�   %            4   +   )         	            �   )            `                        3   "               '   D   �   $         	               {      '         -      8O�<TN:tN[Nć�Py!�O�έP�(OU�hN��7N%��NFxN?�^NĽP�ݔP��N,ςOK��N'/P��N%��N6��N��~N!"�O�e|O�ӫNܨ�PōO�(�M�k�OH�zO��AOI��O��oP/uP1�Oަ[N]5�O(O�Nb��Np�vN�6;O}&Nl�P5;OEk�O��N��N�8eOMTO�TO�B����T���#�
�o��o:�o;ě�<o<o<t�<t�<#�
<#�
<D��<e`B<u<���<���<���<��
<�1<�j<ě�<ě�<�/<�h=o=o=+=t�=�P=�P=�w=#�
=#�
=,1=,1=0 �=@�=]/=m�h=y�#=�%=�+=�+=�C�=�C�=��P=���=�E�=���#0<IU\kmmiX0#���	�����������������������������ypx{}����������{yyyy�������
#5?A<0�����"+/HU_afnrqhaOH</"~wx|���������������~����	"/;=;3( 	��sppptz���������tssss������������������������

�����������YZZ[gktutrg[YYYYYYYY
#-0<GG?<00&#
)7[��������~g[B.jwxpt||�����������tj,16BGMIB86,,,,,,,,,,~������������������~���������������������������):HOH6����srtx�����tssssssssss��������������������������

����	%)/)										��������

��������!#)5Bgruq[A5)!FHUagnz���}zninaWOHF���
#<HSPH9/##�#/<LUgmrndaU</#;8;<@HKMH<;;;;;;;;;;)26>>65)����������������������������������������"#-/<H]inyngaH</("�����5BFEB5)�����������
#%%
�������)5BF@@<6-)��E?GHUXaddaUHEEEEEEEE�����������fns{��������{nffffffsmhlt�����vtssssssss$&$ 	)-/6883)��������������������������6ORD6'����������*6:3*%��:<@HOTamu������mTH;:256BHJBBB62222222222YRSX[hqtt|xth`[YYYY��������

�����������
#%)*(#!
����
�����������
! �x���������������������������x�l�S�X�l�x������������������ù÷ùù����������������)�6�7�6�2�.�)�!������������r���������������������r�p�k�o�r�r�r�r�(�4�=�g�t�w�o�q�n�Z�(���޽ܽ����(�������&�$����	������������������������������������������������s�`�k�s���H�T�^�a�h�n�t�w�s�m�a�T�H�;�/�)�.�/�<�H�zÇÓàäàÛÚÓÉÇ�z�z�v�t�y�z�z�z�z���������������������������������������Ҽ'�4�@�I�A�@�4�'�&� �'�'�'�'�'�'�'�'�'�'�������������������������������������������������������������r�p�f�f�r�t�}�~��O�gĐā�q�L�E�6�)��������ìñ�����;�O�ʾ�	��.�;�G�Q�F�>�;�"���׾˾������ʼʼּڼټּʼ��������ʼʼʼʼʼʼʼʼʼ��/�H�T�[�a�d�i�j�f�e�a�T�K�D�:�4�1�/�,�/�m�y�~����y�m�l�c�j�m�m�m�m�m�m�m�m�m�m�������	�;�H�W�\�W�D�/���������������������������������������������������������N�O�U�Z�^�Z�O�N�M�A�?�<�A�J�N�N�N�N�N�N�y�������������������y�m�`�U�[�`�g�m�x�y�Z�f�j�j�f�a�Z�U�T�V�Z�Z�Z�Z�Z�Z�Z�Z�Z�Z�����(�7�;�;�5�$�������׿Ŀ��Ŀѿ�Ŀݿ������#�������ݿͿĿ��������(�)�,�"�(�1�-�(�������������(�(�����׾�	�"�0�:�?�4�+�����׾����������H�T�a�i�o�o�k�a�W�H�;�/�����������(�H¦«¦¡���	��"�-�"��	����׾˾ʾþƾʾ׾ݾ��Z�g���������������������s�Z�5�(�'�/�N�Z����������������ܹ۹Ϲѹܹ��A�M�f�t����}�u�f�Z�P�A�8�4�3���)�4�A²¿����	�������²�z�{ ¨¨²D�D�D�D�D�D�D�D�D�D�D�D�D�D{DoDfDhD{D�D���0�<�H�M�P�Q�V�U�<�0��
�������������������������������������������������������������!�$�"�!����������׺ں���������������������������������������������������������������������������������������������������������������hƁƚƧƳ����������ƳƚƎ�u�h�]�Y�\�b�h�(�4�7�6�4�(�����(�(�(�(�(�(�(�(�(�(�ûܻ����	��������ܻλ������������ýS�`�l�y�{�{�|������y�l�`�W�S�P�M�M�N�SĚĦĳ������������ĿķĪĦĚĔĄāāčĚ���ɺʺԺɺ��������������������������������'�4�8�@�@�@�@�4�'���������E�E�E�E�E�E�E�E�E�E�E�E�E�EyEuEtEtEuEyE�ǈǔǡǨǭǫǤǡǔǈ�{�p�o�k�o�r�{�~ǈǈ�I�E�G�I�N�U�_�n�{ňŖśśŗŎŇ�{�b�U�I !  @ 3 ; . - T = 0 A ) X H 1 7 e B < & R M c Q C k a H K V h / 0 @ 0 4 = + m 0 I l ? E n < 7 0 A !     �  �  �  �  
  (  �  �  �  F  1  Q    j  �  K  	  P  &  9  x  #  d  H    *  �  I    �  T  �  �        g  g  �    �  g  9  j  �  �    �  �  F  D<�C��o;ě��o=P�`=0 �=<j<���<�t�<�C�<T��<D��<�C�>^5?=aG�<��
=C�<�1=�<���<�/='�<���=D��=H�9='�=���=��=�w=e`B=}�=q��=��w=�"�>]/=��w=49X=y�#=e`B=��=�o=���=�+>@�=�1=��=��P=�E�=��m=�;d>�RB%�OBz�B�jB)GSB#�oB��B�A���B
b�B�AB#��B	,|B%t�B��B 
3B��B�/BEOBAB��Bp�B��B!.B!6B�MB�B�"B��B��B�Bn�BюB%tB�_B3GB�B�BG�B(�4B��B��B]�Bf�B;dB.��A�~�BPB�mB�B@B*�B%ʯB�>BA�B)�"B$��B�BB}�A�x�B
d^BǀB#×B	;%B%�:B�VB�B �BG�B�B�B��B��B��B'�B7BB�B?BB�ZB<AB-SB�iB��B>AB;B?�B�,BM�B>�B(��B�<B��BE�B�jBB�B.�vA�|�B:�B��B�)B4`B��@���A�1{A��@�A7,�A���A�LA��IA���A��@���AH��@��A���AX�0@��A��NAl+-A���A�JCA�lWAmX�A?~,A�0aA��A��$AV�A��A�r�AV�A�x�?3�A=\;A��=C��A��A�I'@Vw�@Z(@�eA��B�A6�@�coA�xAใ@.e�@Ɍ�C��B�cA�@��A�}A�:�@��6A7�A҅!A���A���A�P/A�
:@��AG�j@�|A��4AZf@��SA���Al��A�FuA�uNA�sAm>A?y9A�A�tA�I�AZ��A��A���AU!�A��|?�A=HA�  C��xA�$A�n�@S��@T!S@���A��B>HA6�P@�8�A�A��@,��@�Q�C��B��A�M�   %            5   +   )         	            �   *            a                        4   "               (   D   �   %         
               {      '         -      8   !            5      )                     E   )            9               '   '      +   %         '         ,   '   #                        +                                    /                           !               '               '   #      +   #         '         !   !                           #                     N��KN:tN[Nć�P'SN�UVO�x�N�CN��7N%��NFxN?�^NĽO�8sO~OhN,ςO7��N'/PS�N%��N6��N�e�N!"�O���O�s,N3��P�fOأ�M�k�O?�O��AO0j�O���OՁ�O�ˎO�N]5�O�Nb��NRrON�6;O}&Nl�O�|cOEk�O^̤N��N�8eO%]�O�TO�B    u  d  �  �  M  _    �  �  �  ?  �  �  �  �  �  �  	)  ~    �    5    }    _  �  �  X  T  �  T  A  �  �  �  �  �  �  �    X  �  D  .    	�  �  q:�o�T���#�
�o<#�
<�o<�o<e`B<o<t�<t�<#�
<#�
=�/=o<u<��
<���=T��<��
<�1<���<ě�<���<�`B=o=+=C�=+=�w=�P=�w=#�
=q��=��-=L��=,1=49X=@�=aG�=m�h=y�#=�%=\=�+=��-=�C�=��P=�1=�E�=���# #%00<=IMPLIB<0###���	�����������������������������ypx{}����������{yyyy�����
#+5=<0#������,)(+/<HLUW[UPH</,,,,���������������������	!"//3/*" 	sppptz���������tssss������������������������

�����������YZZ[gktutrg[YYYYYYYY
#-0<GG?<00&#
568>BN[grwz{ytg[NB95��������������������,16BGMIB86,,,,,,,,,,�������������������������������������������)6?A@;3)�����srtx�����tssssssssss������������������������

	������	%)/)										���������

�������)3Bgqtp[N>5)"%$lnqnennz{��|znllllll���
#<HSOH9/""�#/<JUaelpmaU</#;8;<@HKMH<;;;;;;;;;;")/6;<61)����������������������������������������"#/<H\hnvnaUH</)"������09>?<5)�������
  
������������)57:6/)��E?GHUXaddaUHEEEEEEEE�����������fns{��������{nffffffojnt�����toooooooooo$&$ 	)-/6883)��������������������������%-$������������*6:3*%��E@ADHJTamz{��}zmaTHE256BHJBBB62222222222YRSX[hqtt|xth`[YYYY��������


����������
#%)*(#!
����
�����������
! �x���������������������������x�t�m�w�x�x������������������ù÷ùù����������������)�6�7�6�2�.�)�!������������r���������������������r�p�k�o�r�r�r�r�4�A�Z�Y�c�h�c�Z�A������������4�����������������������������������������������������������������y�{�����H�T�a�c�i�m�n�m�m�a�Z�T�K�H�;�9�;�=�H�H�zÇÓàäàÛÚÓÉÇ�z�z�v�t�y�z�z�z�z���������������������������������������Ҽ'�4�@�I�A�@�4�'�&� �'�'�'�'�'�'�'�'�'�'�������������������������������������������������������������r�p�f�f�r�t�}�~���)�6�B�H�O�L�D�6�)�����������������׾��	��"�%�-�*�"��	���׾ԾξξԾ׼ʼּڼټּʼ��������ʼʼʼʼʼʼʼʼʼ��;�H�T�Y�a�c�h�i�e�d�a�T�L�H�E�<�5�3�1�;�m�y�~����y�m�l�c�j�m�m�m�m�m�m�m�m�m�m�/�;�H�I�D�/��	����������������������/����������������������������������������N�O�U�Z�^�Z�O�N�M�A�?�<�A�J�N�N�N�N�N�N���������������z�y�m�b�`�_�`�i�m�y�������Z�f�j�j�f�a�Z�U�T�V�Z�Z�Z�Z�Z�Z�Z�Z�Z�Z�����(�.�6�:�:�5�#�������ٿƿĿѿ������ �������ݿο˿Ŀ����Ŀѿݿ�������(�)�)�(�����������������׾�	�"�.�9�>�3�*�����׾����������H�T�a�f�m�m�i�a�U�L�H�;�/������ ��*�H¦«¦¡����	������	����׾Ͼʾƾɾʾ׾���Z�g���������������������s�Z�5�(�'�/�N�Z������������������ݹѹӹܹ��A�M�Z�f�s��}�u�f�Z�N�A�:�3�)���*�4�A²¿����������������¿²¢«²D�D�D�D�D�D�D�D�D�D�D�D�DzDoDqD{D�D�D�D��
��#�0�8�<�B�E�G�@�<�0��
����������
������������������������������������������������!�#�!����������ٺܺ���������������������������������������������������������������������������������������������������������������hƁƚƧƳ����������ƳƚƎ�u�h�]�Y�\�b�h�(�4�7�6�4�(�����(�(�(�(�(�(�(�(�(�(�ܻ����� ��������ܻû������������лܽS�`�l�y�{�{�|������y�l�`�W�S�P�M�M�N�SčĚĦĳļĿ��������ĿĳĦĚĘĎĊĉċč���ɺʺԺɺ��������������������������������'�4�8�@�@�@�@�4�'���������E�E�E�E�E�E�E�E�E�E�E�E�E�E�E{EwEvEyE�E�ǈǔǡǨǭǫǤǡǔǈ�{�p�o�k�o�r�{�~ǈǈ�I�E�G�I�N�U�_�n�{ňŖśśŗŎŇ�{�b�U�I $  @ 3 :  0 P = 0 A ) X /   7 b B 2 & R D c P 9 M ` K K F h & . < 5 ' = * m 0 I l ? ( n " 7 0 ; !       �  �  �  �    �  	  �  F  1  Q    �  �  K  �  P  o  9  x  �  d    �  [  �      y  T  y  f  �  �  D  g  I  �  S  �  g  9  �  �  �    �  s  F  D  Bz  Bz  Bz  Bz  Bz  Bz  Bz  Bz  Bz  Bz  Bz  Bz  Bz  Bz  Bz  Bz  Bz  Bz  Bz  Bz  Bz  Bz  Bz  Bz  Bz  Bz  Bz  Bz  Bz  Bz  Bz  Bz  Bz  Bz  Bz  Bz  Bz  Bz  Bz  Bz  Bz  Bz  Bz  Bz  Bz  Bz  Bz  Bz  Bz  Bz  Bz  �  �  �  �    
                �  �  �  o    �  �  u  k  \  C  ,    �  �  �  �  �  p  P  .    �  �  �  {  U  d  ~  �  �  �    (  L  q  �  �  �    &  J  l  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  |  �  z  v  h  +  �  b  �     �    o  �    .  @  J  L  H  <  %    �  �  7  �  6  �     \  �    ,  P  ^  ^  V  C  '    �  �  �  �  6  �  �  u  �    �  �  �  �            �  �  �  �  q  C    �  �  \  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  r  W  5    �  G  �  �  �  �  �  �  �  �  |  l  [  K  9    �  �  �  �  w  V  �  �  y  m  `  P  6      �  �  �  �  v  Y  =  !     �   �  ?  <  9  6  3  0  -  *  '  $              �   �   �   �   �  �  �    }  u  m  d  \  S  K  B  :  0  '              	�    �  �  �  �  D  �  �  �  �  �  ~     �  �  �  7  	d    -  {  �  �  �  �  �  �  �  �  �  �  t  0  �  �  S    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  s  X  >  #  �  �  �  �  �  �  l  F    �  �  �  �  f  H  .  �    "  �  �  �  �  �  �  �  �  �  �  �  �  �  w  o  g  ^  V  N  F  =  
  V  d  �  �  �  	#  	$  �  �  �  W    �  �  j  �  	  �    ~  w  q  k  g  c  _  \  Z  X  V  U  S  O  H  A  :  4  -  &          	        �  �  �  �  �  �  �  q  ]  G  2    �  �  �  �  �  �  �  r  M  "  �  �  �  N    �  �  p  �  �                �  �  �  �  �  �  �  �  �  �  �  �  �  ,  4  /  $       �  �  �  �  z  ^  g  :  	  �  �  T  �  �  �          �  �  �  �  �  �  �  `  7    �  �  +  �  b  �    '  F  e  �  �  �  �  �  u  _  B  %    �  �  �  �  �      �  �  �  �  e  *  �  �  �  �  �  ?  �  W  �  0  �  0  U  ^  ^  W  K  9  !    �  �  �  C  �  �  b    �  !  �  '  �  �  �  �  �  �  r  g  _  W  P  H  @  8  0  (       
  �  �  �  �  �  �  �  �  �  e  ?    �  �  �  T     �  �  �    X  S  J  @  2      �  �  �  l  5  �  �  �  M  "  �  �  "  E  Q  R  F  4      �  �  �  y  I    �  v  ,  �  �  �  �  �  �  ~  q  _  F  /    �  �  �  �  y  <  �  �    �  �    �  �    5  I  S  O  ;  $    �  �  5  �  =  �  �  �  �  �  �  k  �  "  @  6    �  d  �  N  �  �  �  �  �  p  I    �  l  �  �  �  �  �  �  �  �  �  ^  &  �  �  L  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  ~  z  v  r  n  �  �  �  o  X  ?  %  	  �  �  �  �  V  $  �  �  s  &  �  p  �  �  �  �  �  �  �  m  P  =  3  '        �  �  �  �  �  �  �  �  ~  h  P  7    �  �  �  i  5    �  �  �  �  �  �  �  �  �  �  �  �  �  �  |  t  j  ]  Q  ?  #    �  �  �  Y  �  �  �  �  �  }  n  [  F  -    �  �  |  6  �  �  k  3            �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  0  �  �  7  S  R  7    �  �    �    
k  	�  �  �  �    �  �  �  �  �  �  �  �  �  q  L    �  �  b    �  �  L  +  �  �    3  8  A  A  5    �  �  �  [    �  e  �  N  �    �  .  )  $           
    �  �  �  �  �  �  �              	  �  �  �  �  �  �  �  c  E  $  �  �  �  j  6  �  �  	i  	�  	�  	�  	�  	�  	x  	d  	L  	'  �  �  ]  �  {  �  n  �  �  0  �  �  �  n  T  8    �  �  �  �  U    �  �  G  �  �  0  �  q  3  
�  
�  
R  	�  	�  	>  �  b  �  �    �  K  �    4  ^  �