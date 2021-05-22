CDF       
      obs    1   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?�ȴ9Xb      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       N�v   max       P��      �  p   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ��1   max       =���      �  4   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?h�\)   max       @FG�z�     �  �   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��z�G�     max       @vG
=p��     �  '�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @+         max       @M            d  /H   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @�>@          �  /�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �D��   max       >p��      �  0p   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A���   max       B-�p      �  14   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A��   max       B,�7      �  1�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       ?9�   max       C���      �  2�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       ?O�   max       C���      �  3�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          �      �  4D   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          G      �  5   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          9      �  5�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       N�v   max       Pp      �  6�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�4�J�   max       ?࿱[W>�      �  7T   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ����   max       >
=q      �  8   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?h�\)   max       @FG�z�     �  8�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?���R    max       @vD��
=p     �  @�   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @+         max       @M            d  H,   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @�o@          �  H�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         C�   max         C�      �  IT   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�:)�y��   max       ?�j~��#     P  J               !   &   	         ]   (            &         G         	      V         &         $      +   �      C   -   U               *         ^         �      O���N���O.�MN�PH/WO���N:��N��NO$C�P�m�P:~%N?�YN�N��9O�E2O�/hN��O�N�+O:�N���O��tP��OJ/`O�M�P%D^N�6�N�O��O���PS
�P��O��O��O{�P8\OriNځ�O.�mN֑KO���O��6N�k�P�?N�vO�_Pw�O>EOas���1�D���#�
�o��`B��`B��`B��o;�o;��
;�`B<49X<49X<D��<T��<e`B<e`B<�o<�o<�t�<�t�<��
<�9X<�9X<���<�`B<�<�=o=o=C�=C�=C�=C�=C�=C�=t�=#�
='�=0 �=8Q�=@�=H�9=L��=e`B=m�h=�%=��=���TR[gt�����������tg[T��������������������~z����������������~njhlt��������tnnnnnn^YVYmx�����������zf^���� 0AJRKB5�����
)6>BNJB6))#*/<=HIJE<9/)# �����(5HVIKF=����0197BO[n������th[?=0��������������������FIKUZbnonb]URIFFFFFF��������������������������
#/8851$
����������#"" ������������������������ebclt�������������jeQMLUaansvqna`UQQQQQQ($! "()5BDJONJNNB5)(�������������������� )1QVWSRRNB5)"���'5NbjL5������jrt���������������tj����
/58AD<+#
�����-5GQB)	����-+//<HMU[\UTH<4/----��������������������Z\\_cjz����������zaZ���� 
#08:71*#
��	.5BN[eeZKEFA5��������
"
����������� 

������mighqz�����������zrm����������������������������
�������������!"$$$$��XQU[gtx}���~tgb[XXXX�����
#*.+(#
�����������������������������������������������$01.%�����GEFHKRTacedcaZTHGGGGB??BO[hny���}t[OIHJB�������������������� )BO[cehnd[OB6)%z�����*44%��������������������������������������������������������������������������������������źú���������������������������Ŀ������������������ĿĳĦĝĢĦĩĳĹĿ�n�zÇÓß×ÓÉÇ�z�n�d�c�e�n�n�n�n�n�n�s���������������������������\�I�H�Z�s�[�h�t�|ěĦĭĦĚčā�h�[�V�W�_�_�Y�U�[ìù������������üùìéììììììììÇÓÕÕÖÓÑÇ�z�y�r�z�zÄÇÇÇÇÇÇ�T�a�l�v�z�~�{�z�o�m�a�T�I�H�=�;�2�;�H�T�0�I�O�[�mňŔő�{�b�0����������������0������ʾѾѾ;����������Z�H�F�C�D�Z�f����������������������������������������伤���������������������������������������N�W�[�]�[�Z�R�N�B�5�-�/�5�8�B�C�N�N�N�N����)�6�=�<�5�,�)������������������"�;�G�T�`�j�m�n�f�G�;�.�"�	������	��"����	���	�����������������6�B�O�[�b�o�q�l�\�6����� ��� ��)�6���)�4�)�(� �����������������ʾ׾������������׾ʾ������������f�s�����������������z�s�i�f�f�f�f�f�f�G�T�`�m�y�������y�m�`�T�G�;�3�$�%�0�;�G�z���t�u�\�;�"��	����������������;�T�z�	���-�;�=�E�A�?�;�/�"���	�����	�`�m�����������y�`�T�;�&�"��"�.�;�G�[�`��(�A�P�R�N�A�7�(����߿Ŀ����ĿϿ�������������������������������������뽅���������ĽʽǽĽ��������������z�������[�hāčĚĦĮĭĩĦĚčā�t�l�h�]�V�[�[�f�����������������s�f�Z�M�B�D�J�M�Z�fƧ���������0�;�>�=�*����ƧƁ�u�\�J�uƧDoD{D�D�D�D�D�D�D�D�D�D�D�D�D�DrDdDdDiDo�/�<�F�H�U�a�a�U�H�F�<�/�#������#�/��(�5�A�N�R�W�X�W�Y�N������������������%�*�#�������ܹϹ������ù׹���������������������������n�f�a�X�Z�i����!�-�:�S�U�`�_�X�S�F�:�-��������ݽ�������������ݽҽнʽ˽нٽݽݽݽݾ4�A�M�Z�d�s�{������s�f�Z�L�A�;�4�2�,�4�����ʼҼּ׼׼ּּʼ¼�����������������������������ùêà×àäíù�������y���������������������y�l�h�f�^�_�Z�l�y��#�0�7�<�I�R�P�I�<�0�#� ��������r���������¼������r�M�4������'�M�rÇÓÓÛÖÓÇÆ�}ÁÇÇÇÇÇÇÇÇÇÇ�r���������ԺԺǺ��������������r�f�b�e�r��'�=�G�I�@�7�.�����ܻŻ����ʻ������"�(�5�@�A�O�X�N�A�5�(������������
��#�8�9�2�&�#��
������������������ I 1 ! : F S d b , 2 0 I h H  3 3  V ? ; / h W P ) + R -  t : = - R 3 ' 2 O 7 V V U h j J ] h .    "  �  l  �  �    t  �  h  �    \  r  �    b  �    �  �  �  1  �  �  �  �  �    /    �  v  :  �  :  �  �  �  �  �      �  8  w  �  �  o  ǻD��;��
;��
<t�<���<��;o<D��<�t�=ȴ9=@�<T��<T��<��
=L��=,1<�o=�-<�9X=C�<�/=<j=�"�=49X=@�=�+=T��=49X=�C�=L��=���>Y�=D��=��=���=�F=��=49X=e`B=]/=� �=�hs=y�#>t�=}�=�->p��=�-=�`BB
#B"8�B=B
6fB �oB?�B�^B��B�B�OB �B�	B'jB�<B�#B˂B!�>B
�=BA�BIB��BO�B�B+�Bn�B��B�VB �B �BD)B��B�B.�BfBc�B9	B��B	k]B$�_B"�\B5�B-�pA���B/�B!�QB$�B��B�$B��B	KB"@B@�B
<�B ��B�)B?mB�BA�B��B�UB�B'W�B�TB��B��B!�iB?QB<B@�B�^B=�BB�B<�BDyBɽB�B B�B >NB@�B?�B>�B�BJsB@B;�B��B	x�B$w]B"¢BA"B,�7A��B��B!�YB�EB°B�nB�UA��@J�A��A���A��A�lZA�4/A�S�A�7�A��AE�JA��*@��OA��A��yAbesAY:TA���A�-#ATPIAEAg��A�ArA�s�AhV�A�|�A�)�A!��A��qAB�BDvC���A��A��Z?9�A�t@v�"A,6�A>�a@��eAИ�AE�A��@�ZRAɹ�@�@�]�A�'A��eA�}@��A�{�AȀSA��A܎�A�U�A�~AA�x�A�^�AE A�b�@�ߩA��A��AbrAY�AׁA�,AT�^ACEQAh�A���A���Ak�A�q�A�vxA"M�A�y�AAiBO�C���AA��?O�A�5@zYIA,�2A>�7@�JyA��A��A�W@���Aɓ�@
C�@��A�OkA��i               "   '   
         ^   )         	   &         H         
      V         '         %      +   �      D   .   V               *         _         �                     /   #            ;   +            !         !               G      !   /               7   #            )               %   !      +      !   3                     %                                                      9         )               7               !                  !      %      !         O]RN��-O�N�O��HOr_N:��NB-�N�sO�a�O�F�N?�YN�N��9Oj�OFl�N��OvYjN�+N�#N���O[�6PpOs�O��zP\N���N�O>FO���PS
�O��O��O���O&��O�
SOc�Nځ�O��N֑KO�q�O��6N��Oɮ`N�vO�*zO�Nы�Oas�  �  �  [  q  �  6  g  7  �  �  �  v  �  �  f    P  �  ~  p    �  �  ?  3  �  �  �  �  �  ,  r  �  	�  �  	�  �    �  _  �  �  �  +  7  %  �  �  м���o�o�o;D��;�o��`B:�o<o=P�`<���<49X<49X<D��<���<�9X<e`B='�<�o<�9X<�t�<���=#�
<���<�h<�=\)<�=#�
=o=C�=�-=C�=49X=49X=P�`=�P=#�
=,1=0 �=L��=@�=L��=�hs=e`B=q��>
=q=���=���WTY[gt����������tg[W��������������������~������������������~njhlt��������tnnnnnnhecbdo~����������zmh���$)58BHGB5����
),69BKCB6-)"#/<AD?</#����)/4685)���LKKQ[ht�������thb[QL��������������������FIKUZbnonb]URIFFFFFF�������������������������
#&)'#
��������������������������������omrt�������������utoQMLUaansvqna`UQQQQQQ'%$&)05<BEKIDB5)''''��������������������#)BGNRSPNNMB5)&���&5W[ND5�����xv|����������������x���
*/344>@</#
�����+5DN?5)������//3<HHTSHC<:1///////��������������������lgeglmuz���������zml���� 
#08:71*#
��	.5BN[eeZKEFA5�������

������������ 

������mkmz������������zwqm������������������������������������������ !#$#"��XQU[gtx}���~tgb[XXXX�����
#%,)'#
�����������������������������������������������$01.%�����EFHLSTabdcbaYTIHEEEEGDGO[hs�����th[WQOG�������������������� )BO[bdhmc[OB6)&�����������������������������������������������������������������������������������������������������������������������������������������������ĿĳĦģĦĨĮĳĿ�����n�zÇÓß×ÓÉÇ�z�n�d�c�e�n�n�n�n�n�n�s�����������������������������g�[�Y�g�s�h�tāđěğĚĘďčā�t�h�_�]�e�f�d�e�hìù������������üùìéììììììììÇÒÓÔÓÎÇ�z�z�u�z�~ÇÇÇÇÇÇÇÇ�T�a�c�m�n�u�r�m�a�T�P�I�I�S�T�T�T�T�T�T�#�0�<�C�L�P�P�H�<�0�#���������������#���������������������f�`�Z�V�X�Z�c�s����������������������������������������伤���������������������������������������N�W�[�]�[�Z�R�N�B�5�-�/�5�8�B�C�N�N�N�N������,�/�,�)� ��������������������"�.�;�G�T�]�`�b�`�^�T�G�;�.�"��	�
��"����	���	�����������������6�B�O�P�Y�^�`�Z�O�B�6�&������)�+�6���)�4�)�(� ���������������ʾ׾������ �������׾ʾž��žʾʾʾʾf�s�����������������z�s�i�f�f�f�f�f�f�T�`�m�y���{�y�s�m�`�T�H�G�;�-�.�7�;�G�T�;�T�a�d�Z�G�/�����������������������;�	��"�)�/�9�;�B�=�;�7�/�"����	���	�m�������������y�m�`�T�;�.�'�&�;�G�R�a�m��(�A�M�O�K�A�4�(������ҿҿ�����������
�����������������������������������������ĽʽǽĽ��������������z�������h�tāčĚĜĦĩĩĦĥĚčā�s�h�f�`�f�h�f�����������������s�f�Z�M�B�D�J�M�Z�fƧ���������0�;�>�=�*����ƧƁ�u�\�J�uƧD�D�D�D�D�D�D�D�D�D�D�D�D�D�D�DwDwD{D�D��/�<�F�H�U�a�a�U�H�F�<�/�#������#�/�(�5�A�M�T�T�R�N�A�(������������(��������$��������ܹϹ̹ιܹ�����������������������������z�p�l�n�s������!�-�:�S�T�_�^�V�S�F�:�-�!�������ݽ�������������ݽҽнʽ˽нٽݽݽݽݾ4�A�M�Z�b�s�z������s�f�Z�M�A�=�4�3�-�4�����ʼҼּ׼׼ּּʼ¼������������������������������������ùñëíòù���ҽy���������������������y�l�h�f�^�_�Z�l�y�#�0�5�<�I�M�L�I�<�0�#�!�����#�#�#�#�r�������������r�f�Y�M�@�0�����'�M�rÇÓÓÛÖÓÇÆ�}ÁÇÇÇÇÇÇÇÇÇÇ���������ҺѺź��������������r�f�c�e�r������%�'�(�&�"�������׻ջػݻ������(�5�:�A�H�A�5�(�"������������
��#�8�9�2�&�#��
������������������ J %  : K Q d ] :  ( I h H  , 3  V 0 ; + ] J [  $ R %  t 2 = " @ ) ( 2 L 7 R V S h j J E E .    �  ~  8  �  }  $  t  {  �  �  ,  \  r  �  �  �  �  �  �     �  �  c  ^  O  |  �    �    �  8  :  u  r    �  �  r  �  q    �  E  w  s  �  �  �  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  |  �  �  �  �  �  �  �  }  y  y  w  h  Q  1    �  �  g  �  a  |  �  �  �  �  �  �  �  w  X  0    �  �  �  [  .  �  >  M  W  Z  Y  U  O  H  @  7  -    
  �  �  �  d    �  S   �  q  k  d  ]  U  L  B  8  .  $    �  �  �  c  "  �  �  9  �  �    J  p  �  �  �  �  z  ]  4  �  �  e    �  [    �    �      $  .  1  4    �  �  �  g  (  �  �  �  8  e  �  e  g  _  X  R  N  E  ;  5  1  (                    �  "  #  +  6  ,      �  �  �  c  0  �  �  �  M    �  �  K  �  �  �  �  �  �  �  �  �  �  �  �  �  �  w  V  *  �  �  �  �    0  >  ;  ;  J  �  �  �  �  �  �  c    �  �    �  �    A  v  �  �  �  �  �  �  �  �  �  �  �  Z    �  :  �  w  v  p  j  d  ^  X  R  L  F  @  9  2  +  $             �  �  �  �  �  �  �  �  �  �  �  t  a  N  ;  (       �   �   �  �  z  p  e  X  K  ;  *    �  �  �  �  x  N      �   �   �   Y  �  �  $  H  X  `  e  f  d  `  Y  L  9    �  �  w    �  �  �  �  �  �  �  �     �  �  �  �  �  �  �  \  &  �  O  �  9  P  L  G  B  >  9  4  0  +  &  "              	      �  )  r  �  �  �  �  �  �  �  �  \    �  E  �  �  �  c  +  ~  y  t  o  d  Y  M  @  2  $      �  �  �  �  �  �  �  �  E  W  e  l  o  o  k  b  T  B  ,    �  �  �  [    �  c   �    �  �  �  �  �  �  �  o  ]  L  3    �  �  �  �  �    G  �  �  �  �  �  �  �  �  �  �  �  \  -  �  �  d  �  u  �  	  �    y  �  �  �  �  V  +  /  N  '  �  V  �  G  �  �  c  R  >  9  ;  ?  =  5  #    �  �  t  6  �  �  b    �  N  �  =  *  +  .  2  3  '    �  �  �  e  F  X  !  �  �  (  �  a   �  �  �  �  �  �  �  �  �  �  �  z  P  !  �  �  W  �  �  1  5  P  u  �  �  �  �  �  �  �  �  m  S  5    �  �  �  x  J    �  �  �  �  �  v  U  0  	  �  �  �  ]  2    �  �  |  M    �  �  �  �  �  �  �  �  �  �  Y  "  �  �  <  �  =  �  �  �  �  �  �  �  |  j  W  C  -    �  �  �  �  y  D     �   �   |  ,      �        �  �  �  �  �  a  '  �  �  �    *  �  �  �  i  �  5  h  n  \  ,  �  l  �  �  �    �  �  "  
L  f  �  �  �  �  �  �  �  �  t  i  ^  T  G  6  #    �  �  �  4  	a  	t  	�  	�  	|  	d  	>  	  �  �  F  �  �  U  �  e  �  �    �  �  R  {  �  �  �  �  s  [  4  �  �  U  �  F  �  �  �  �   �  �  	_  	~  	�  	�  	�  	{  	K  	  �  x    �  F  �  9  �  �  �  
  �  �  �  �  �  �    j  S  5    �  �  {  2  �  V  �  �  <      �  �  �  �  �  �  �  �  �  �  y  h  W  ?  %     �   �  �  �  �  �  �  z  �  y  h  R  8    �  �  �  �  l  3  �  �  _  W  O  E  :  0  $      �  �  �  �  �  �  �  a  $  �  �  B  �  �  �  y  n  �  �  �  a  *  �  �  R  �  z  �  -  G  E  �  �  �  �  �  �  �  `  @  &    �      �  �  �  N  �   �  �  �  �  �  �  �  �  �  `  /  �  �  �  [  !  �  �  n    �  �  N  �  
  *  #    �  k    �    
�  	�  	1  2  �    V  +  7  /  '        �  �  �  �  �  _  =    �  �  �  �  n  J  $      �  �  �  �  �  ^  /  �  �  w  ,  �  �  @    �    �  �  U  �  �  $  N  g  �  e  .  �  X  �  �  �  
�  	7  �    t  `  |  �  �  �  �  �  �  q  [  @  #    �  �  �  o  8  �  �  �  �  �  t  L    �  �  g  !  �  �  4  �  �  ,  �  �  @