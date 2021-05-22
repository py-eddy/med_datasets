CDF       
      obs    6   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?�Ƨ-      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M҃�   max       P���      �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       �T��   max       =���      �  \   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�p��
>   max       @E,�����     p   4   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�         max       @vg�z�H     p  (�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @+         max       @P�           l  1   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @˼        max       @�y           �  1�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ���
   max       >�+      �  2X   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�b2   max       B(��      �  30   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�~�   max       B(�[      �  4   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       @	)L   max       C��      �  4�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       @��   max       C��      �  5�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          �      �  6�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          =      �  7h   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          7      �  8@   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M҃�   max       P�8,      �  9   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�N;�5�Y   max       ?��f�A�      �  9�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       �T��   max       >hs      �  :�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>��Q�   max       @E,�����     p  ;�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�         max       @vg
=p��     p  D   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @(         max       @P�           l  L�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @˼        max       @�7           �  L�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         EW   max         EW      �  M�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��-V   max       ?��f�A�     �  N�      M      .      J               
         ]      
      $      V      V            G      P   -            	            =      �   %                     "   !            >   S   N)MVP�$N@!�P\��N��P��*O/� NXh�O8;N���N�M�N�>N�S�P��O���O�`N���Pb�O<Q�P���N�"�PV��N6	.N��O��P@��N�z�O��P��O
��NMl�M҃�N5KO�b�O�H�Nfp�O变NbE�O�mOH�Ni��NLF�N�N��0N-�N6l6O_��Ot�|OW�"N9n\O)1Ox+�O�j�NW{��T����o�D����o;D��;D��;D��<t�<t�<#�
<T��<u<�C�<�C�<��
<��
<��
<�j<�j<���<���<�`B<�`B<�<�=o=o=+=C�=C�=C�=t�=t�='�=49X=49X=8Q�=@�=D��=T��=]/=ix�=u=y�#=�%=�7L=�7L=��
=��
=�{=� �=��=\=���~y��������~~~~~~~~~~�����
/<HOOCC?/#���������������������������H_raU<32-#
��	".&"	�����<NUYXNB5�����������

��������������������������*+-/8<HU[aefaXUH<1/*�������������������������

��������zxz�������������������
����������)BSY\[PB5)������������������������)/68:96)";77=BGOPUXWOOB;;;;;;mhjq��������������vm������������
������)[������[B5)##-+(##����6D[h���h[B)����������������������������������������������������������������)6IXfkf_O6�������������������������#'0>EEA90#
��)5[t���|tb[NB5) �����������������������

�����������������������������nnz{�����{nnnnnnnnnn���������������)6BPQ^O6+��#(0<60#����������������������������������������������
  
���������������������������������������������.(%'/0<<>A</........����

�����������" "$*/1;;?AA=;6/*"""506BO[\[TOB655555555?9BNNO[d[YNB????????^\^cgt���������tqhh^]UY_amz���������zma]������),'��������������������������������������������

�����������������#%,%#"�n�zÅÃ�z�o�n�c�j�g�n�n�n�n�n�n�n�n�n�n��(�N�s�����������������g�A��ۿҿ����"�.�1�.�"�������	� �������	���"�/�T�z���������������r�l�U�;�"�	��
��/�H�T�[�a�e�d�a�T�N�H�@�@�H�H�H�H�H�H�H�H��<�I�{ŎŞŞ�n�S�<�#�
������������������������������������������w�n�p�x�z�������ʾξоʾž�������������������������������������������������������������(�5�A�B�N�Z�^�f�f�Z�N�D�A�5�1�(�#�"�(�(��������
������������������������������������	�������������������&�*�6�;�9�6�*�"����������Ƨ�������$�"��������ƧƇ�r�_�C�I�hƧ�/�;�H�T�[�c�`�W�H�;�/�$�� �	����"�/�	���"�$�����	������������������	�z���������������������z�z�s�z�z�z�z�z�z�(�4�A�M�f����������s�Z�4�������(�r�������������������������z�r�m�j�q�r��/�H�T�Y�V�?�5�/��	���������������	��B�N�[�g�t�z�t�p�g�[�N�B�A�<�B�B�B�B�B�B�ʼ�����ּܼü¼����������z�v�����������/�<�H�D�<�3�/�-�+�#�/�/�/�/�/�/�/�/�/�/²¿������������¿º²¦£¤¦¯²²²²�A�M�Z�d�f�h�a�Z�M�A�4�(����'�(�4�5�A�׾���6�P�;�0�,�"�	��ʾ���������������àëìùû��ùøìàÓÌÌÎÓÔàààà�����ɺ���-�(�!������ɺ������������`�m�������������y�m�T�F�;�8�6�7�:�D�T�`�������������ĿȿĿ����������������������A�M�T�T�S�M�A�4�4�4�4�?�A�A�A�A�A�A�A�A���������������������������������������������������������������������������#�/�<�H�I�R�S�O�H�<�/�#�������
���L�Y�r�����������ͺպ��������m�e�V�U�F�L��������������ۼ׼����������āĚĦĵĿ��ļĳĦĚčā�[�F�L�J�O�\�sā���(�/�4�8�?�A�D�C�A�>�4�(�$�����D�D�D�D�D�D�D�D�D�D�D�D�D�D~DuDrDuD}D�D��o�{ǈǎǘǡǢǟǔǈ�{�o�j�b�]�X�[�b�g�o����������������������������������;�G�T�`�a�`�T�T�G�;�0�7�;�;�;�;�;�;�;�;�h�tĀāăā�t�h�f�d�h�h�h�h�h�h�h�h�h�h�#�0�9�<�I�U�I�<�<�0�#��
��
���#�#�#�
��������
���	�
�
�
�
�
�
�
�
���������������������)�6�B�O�S�U�M�?�6�.�)���������)ŠŭŹ������������������ŹŬťŠŝşŞŠ�B�N�[�g�r�t�~�t�g�Y�P�M�I�B�<�B�	�	�����������
�����	�	�	�	�	�	�	�	����������������������������������������E�E�E�E�E�E�E�E�E�E�E�E�E�E�E{EuEjEjE�E������лܻ������ܻ������l�_�Z�a�l�����b�n�{ŃŇňŇ�{�n�b�b�_�b�b�b�b�b�b�b�b c ] � Q I 1 . D & Z @ + 2 C # @ 6 ( ; ; L k Z a C C # 8 # 1 D l K  p 9 , ] ( F 8 a F X p R 3 P e v > ' a 0  E  �  �  �  �  �  r  _  X    �  �    �  `  c  �  �  �  1  �  [  �  �  K  e  �  I  M  =  b    O  
  Q  |    �  �  �  �  ^  8    �  ]  �  '  �  �  {  �  �  p���
=��%   =49X<o=��-<�/<T��=�P<�9X<�j<�h<�9X=�;d='�<�h<��=m�h=,1=�S�=o=�x�=+=L��=e`B=��=L��=�`B=���=L��=�P=��=8Q�=�hs=�hs=P�`=�
==]/>�+=�E�=y�#=�%=�\)=�7L=��=��P=��=�l�=��`=�Q�=�l�>�w>6E�=�l�B
�dB��B a�B�EA�b2B�B#!�B�BaB��Bn�BNB�HBGvBɆB�9Bi�B�B"�;B��Bs�B�B��B"~B�B�)B!�B%B7mB��B$E�B"�@B(��B�/B�B%Z�B 6B t�BEBtBBkB�EA��
Bd�B7�B
9B �BGB�B��B�B{BrB
G7B�[B�rB�:A�~�B?B#8�B�B׊B8B@�B=QB�AB��B��B�B�nB�B"��B�B�B�B�cB>�B�fB�0B"/B$ƘB�gB�$B$]�B"CB(�[B>�BBB%UOB�*B @7B>NB�kB�dBEVBS�A�� BBWBöB	ğB BB�]B��B��B��B�B��A��
A��A]wEA�h�A�d�A�M@���AN��A�:�A���A��L@���A�e�B�3A�8zA���A��A;�a@��A�6A�p�@�&A¢�A��A<��AT�A˹@@�JAj��As��A;�kA <@Z��A��@	)LA<EA��FA8C���BmA���Ae�*A�|^A�#A�GkA�6�A���A��SA��A�i�A��WC��@��LA��xAȀA���A_A���A���A�q@��'AN��A�v�A�cA�s�@�[A�{�B@�A���A�JA���A<Z@�A��A���@�ŻA���A�{YA<��ASkA˅�@D{Aj�At��A;*A ��@[�A�xR@��AMA�lwA8wVC��B�A�~�Ae>A�}AA�A�n�A�wgA׃A��]A��A��A��C��@��4A�    	   N      .      K                        ]      
      $      W      W            H      P   .            
            =      �   &                     #   "            ?   T         9      3      7                        ;            )      =      3            5      %   %                  )      %                                                '         -      -      7                        /            #      5      %            '         !                  )                                                      !   N)MVP*q�N@!�P9iNH�JP�8,N���NXh�N�c�N���N�M�Ne�'N�S�P^/�O�^KO�`NNNO�'�O<Q�P�R�N�"�O��pN6	.N�LN�O�PqN�z�OťO�toO
��NMl�M҃�N5KONxO�H�Nfp�O�\�NbE�O��OH�Ni��NLF�N�N��0N-�N6l6O-/Ot�|OW�"N9n\O)1OIJ@O�gNW{�  |  &  R  �  �  �  �  �  k  T  �  =  �  �  �  �  J  Y  �  	0  X  	    >  /  '  [  Q    �  �  S  s  �    �  	t  /    �  �        �  �  �  �    �  o  �  �  y�T��<�o�D��<t�;�o;�o<D��<t�<�t�<#�
<T��<�C�<�C�=o<�1<��
<�9X<��<�j=\)<���=L��<�`B=+=C�=D��=o=q��=�P=C�=C�=t�=t�=,1=49X=49X=}�=@�>hs=T��=]/=ix�=u=y�#=�%=�7L=�t�=��
=��
=�{=� �=��`=���=���~y��������~~~~~~~~~~����	#6;<8;83/#�������������������������
#/HUZUH</,#
���	"(#"	������;NUYXNB5�������������������������������������������:0/4<HOUXUUH@<::::::�������������������������

��������|z|���������||||||||���������
���������)5NUTPIB5)������������������������)/68:96)"=9;ABFOVVOLB========ztt{���������������z������������
������	5[����~[NB5.	#-+(##�����6EOWYRB6)���������������������������������������������������������������	6BMT^a_XOB6�������������������� #07:970)#
")5BZt|�|yt\NB5)"�����������������������

�����������������������������nnz{�����{nnnnnnnnnn��������������)6BPQ^O6+��#(0<60#�����������������������������������������������


����������������������������������������������.(%'/0<<>A</........����

�����������" "$*/1;;?AA=;6/*"""506BO[\[TOB655555555?9BNNO[d[YNB????????c_`fgty���������tlgc]UY_amz���������zma]������),'�������������������������������������������	

��������������������#%,%#"�n�zÅÃ�z�o�n�c�j�g�n�n�n�n�n�n�n�n�n�n��(�A�Z�s�������������g�A������������"�.�1�.�"�������	� �������	���"�T�m�z�������������{�t�c�H�/����)�;�T�H�T�Y�a�d�b�a�T�S�H�C�C�H�H�H�H�H�H�H�H��<�I�b�{ōŝŜ�n�R�<�#�
�������������������������������������������������������ʾξоʾž����������������������������������������������������������������(�5�A�B�N�Z�^�f�f�Z�N�D�A�5�1�(�#�"�(�(��������
�������������������������������������������������������������&�*�6�;�9�6�*�"���������������������������ƧƖƁ�q�i�jƁƧ���/�;�H�T�Z�_�b�_�V�H�;�/�'������"�/�	���"�$�����	������������������	�z�����������������|�z�u�z�z�z�z�z�z�z�z�(�4�A�M�S�y���~�x�s�f�Z�A�(������(�r�������������������������z�r�m�j�q�r�	��/�;�M�R�L�:�/��	�����������������	�B�N�[�g�t�z�t�p�g�[�N�B�A�<�B�B�B�B�B�B���ʼּ߼�ټҼʼ������������~�����������/�<�H�D�<�3�/�-�+�#�/�/�/�/�/�/�/�/�/�/²¿������������¿³²°¦±²²²²²²�A�M�Z�a�f�_�Z�M�A�4�.�-�4�;�A�A�A�A�A�A�ʾ׾��"�%���	����׾���������������àëìùû��ùøìàÓÌÌÎÓÔàààà���ɺֺ������������ֺ������������m���������������y�`�T�K�;�8�9�=�G�T�`�m�������������ĿȿĿ����������������������A�M�T�T�S�M�A�4�4�4�4�?�A�A�A�A�A�A�A�A���������������������������������������������������������������������������#�/�5�H�Q�R�O�H�<�/�#��������
���L�Y�r�����������ͺպ��������m�e�V�U�F�L��������������ۼ׼����������āčĚġĦĭĵĹĹĳĦčā�t�d�c�h�m�tā���(�/�4�8�?�A�D�C�A�>�4�(�$�����D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D��o�{ǈǎǘǡǢǟǔǈ�{�o�j�b�]�X�[�b�g�o����������������������������������;�G�T�`�a�`�T�T�G�;�0�7�;�;�;�;�;�;�;�;�h�tĀāăā�t�h�f�d�h�h�h�h�h�h�h�h�h�h�#�0�9�<�I�U�I�<�<�0�#��
��
���#�#�#�
��������
���	�
�
�
�
�
�
�
�
���������������������)�6�B�O�P�R�O�J�<�6�)�����
���$�)ŠŭŹ������������������ŹŬťŠŝşŞŠ�B�N�[�g�r�t�~�t�g�Y�P�M�I�B�<�B�	�	�����������
�����	�	�	�	�	�	�	�	����������������������������������������E�E�E�E�E�E�E�E�E�E�E�E�E�E~EvEoEuE�E�E����лܻ�����ܻû������x�m�x���������b�n�{ŃŇňŇ�{�n�b�b�_�b�b�b�b�b�b�b�b c ^ � Q I 1 & D " Z @ , 2 9 $ @ * ' ; 6 L O Z R ) 4 # 1 ! 1 D l K  p 9 $ ]  F 8 a F X p R @ P e v > $ O 0  E  M  �  �  s  x  �  _  �    �  k    �  ,  c  d  �  �  v  �  �  �  �  �  I  �  �  	  =  b    O  �  Q  |    �  D  �  �  ^  8    �  ]  �  '  �  �  {  �  �  p  EW  EW  EW  EW  EW  EW  EW  EW  EW  EW  EW  EW  EW  EW  EW  EW  EW  EW  EW  EW  EW  EW  EW  EW  EW  EW  EW  EW  EW  EW  EW  EW  EW  EW  EW  EW  EW  EW  EW  EW  EW  EW  EW  EW  EW  EW  EW  EW  EW  EW  EW  EW  EW  EW  |  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  b  �  q  F  �  �      &      �  �  �  y  8  �  �  $  �  ,  v  /  R  U  X  Z  ]  `  c  b  _  [  X  U  Q  N  I  E  A  =  8  4  �    K  m  ~  y  Y  ,  �  �  !    �  �  �  �  S    �   �  u  |  �  �  �  �  �  �  �  �  �  �  �  �  �  y  s  l  e  _  �  �  s  U  :  $  	  �  �  �  w  T  ;  $    �  �    2  �  �  �  �  �  �  �  �  �  �  �  �    X  .     �  �  g  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  y  n  c  Y  Q  �  �    3  R  i  f  X  A  '    �  �  �  b  $  �  �  m  T  L  D  9  *      �  �  �  �  �  �  �  |  b  B    �  x  �  �  �  �  �  �  �  }  k  Y  B  '  	  �  �  �  l  6   �   �  <  <  <  ;  4  -  #      �  �  �  �  �  w  w  �  �  �  �  �  �  �  �    x  p  h  a  [  T  M  E  <  4  +      �  �  �  �  �  �  �  �  �  �  �  �  �  �  H  �  m  �  N  x  �  K  �  �  �  �  �  �  �  �  �  u  ^  @    �  �  d  
  �  @   �  �  �  �  �  �  �  �  �  n  W  =  "    �  �  �  {  U  :    =  A  D  H  M  T  Z  `  f  k  p  r  t  r  m  Z  M  G  N  Z    '  <  M  X  W  Q  F  6    �  �  �  d  4    �  z    �  �  �  �  {  n  ]  D  %    �  �  �  k  7  �  �  �  O  B  /  �  	  	0  	*  	  �  �  {  ?    �  �  4  �  �  C  �  �  �   �  X  V  T  P  I  B  :  2  *  "      	    �  �  �  �  �  �  q  �  �  v  �  	  	  �  �  �  [    �  D  �  ?  �  �  �  �        �  �  �  �  �  �  �  �  �  �  �  �  x  H    �  �  �  �    =  4  #          �  �  �  �  �  �  {  P    �  �  %  +  /  -  '      �  �  �  u  7  �  |  �  P  �  �  H  x  �      '  !    �  �  �  c  3  �  �  Y  �  X  �  �  2  [  H  4    �  �  �  s  D    �  �    B    �  �  �  ^  �  #  |  �  �  -  H  P  O  I  ;  &    �  �  B  �  <  �  �  �            �  �  �  �  �  �  �  O    �  p    }  �  m  �  �  �  �  �  �  }  c  F  (       �  �  �  ?  �  �    �  �  �  �  �  z  q  g  _  Y  R  L  E  ?  8  0  (           S  O  K  G  D  @  <  8  4  0  )           �  �  �  �  �  s  ]  H  *    �  �  �  �  E  �  �  x  A    �  �  [     �  �  �  �  t  _  H  0    �  �  �  e  *  �  �  D  �    E  v    �  �  �  �  �  �  �  h  6    �  �  b    �  9    �    �  �  �  �  �  �  �  �  �  �  �  �  �  t  b  I  1  	  �  �  5  �  �  	/  	U  	p  	q  	]  	8  	  �  x    �  :  �  �      t  /  ?  O  _  l  x  ~  }  }  {  x  u  q  n  j  f  a  K  -    �  �       �  r  �      	  �  A  )  �  k  �  |  �  �  2  �  p  g  ]  U  H  4    �  �  �  P  �  �  -  �  6  �    �  �  �  �  w  e  R  :      �  �  �  �  l  @    �  �  N          �  �  �  �  �  �  x  `  G  .    �  `    �  �  ?       �  �  �  �  �  �  �  u  `  K  5       �  �  �  �  �    �  �  �  �  �  �  �  �  q  k  q  v  y  y  y  x  o  f  ]  �  |  v  o  i  b  \  U  O  H  N  `  r  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  v  _  F  -    �  �  �  �  �  �  �  �  �  �  �  �  �  �  q  <  �  �  O  �  i  �    �  r  �  �  ~  `  T  H  ,    �  �  �  T  "  �  �  B  �  m  �  �    �  �  �  �  s  H  !  �  �  �  G  �  �  �  Z    �  A  �  �  �  �  �  v  e  T  C  3  &        �  �  �  �  �  �  �  o  a  R  >  &    �  �  �  P    �  {  )  �  ^  �  F  �  +  �  �  �  �  �  �  �  U  $  
�  
�  
_  	�  	a  �  �  �    g  u  �  t  �  �  l  ]  <  �  �  W  
�  
w  	�  	8  {  �  �  �  !  w  y  P  4  "  �  �  �  �  �  `  9    �  �  x  �  �  \  �  .