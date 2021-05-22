CDF       
      obs    4   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?�9XbM�      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M��   max       P�J      �  |   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ��j   max       ;�o      �  L   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>}p��
>   max       @F.z�G�            effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��z�G�    max       @v��z�H        (<   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @&         max       @N�           h  0\   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�_        max       @��          �  0�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �	7L   max       �49X      �  1�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�H�   max       B/��      �  2d   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A��   max       B/��      �  34   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >\�   max       C��^      �  4   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >Go   max       C��I      �  4�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          P      �  5�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          M      �  6t   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          C      �  7D   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M��   max       P��      �  8   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?���
=p�   max       ?Ѯ�1���      �  8�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ����   max       �o      �  9�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>}p��
>   max       @F"�\(��        :�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?��Q�     max       @v��z�H        B�   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @&         max       @N�           h  J�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�_        max       @��           �  K,   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         A�   max         A�      �  K�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�ݗ�+j�   max       ?Ѯ�1���     @  L�   H   A   &      O            <      =                        !                  $      %                     >      H                           "            
      .   'P���O���P��O��P�JO���Oď�O+zO�?NC4P+�N�x�M��O	5�M���N�N��fP!cO�)UNՆNT�dOBeIO��O��RO��BN�~O��^N]��P��N{ �N�@BN��O��P;��ON^nPY�~N?VTN�:�O��O2A�OT��O2J=N��O��<Ou#�Nb�O72�Ov�bO7�=N�OpUO��;�o;�o;o��o��o�o�o�D����o�o�49X�D���e`B��o��C���C����㼬1��9X���ͼ��ͼ���������/��`B��`B��`B��h�o�o�+�+�\)��P��P��P����w�49X�@��D���D���L�ͽT���Y��Y��aG��m�h�m�h��O߽�9X��jam��������
����kfa��������������������)BFX\jmyp[OB6)S[gt����������k`TOOS��	#<U|������{U<0��������������������>HUWnr�����zaUNHA:8>knz������zqnljiecdfk��������������������'06<>AA<90-%''''''''/HUih���na\U</#������

������������������������wz�����������{zytqww��������������������EHJUaeimnqnjaUKHEEEE�����		
	������������������
 �������)+*6BLh���tmh[OB1%!)

#//43/*#
)-24)!���������������������*6CMTSOC*�������������������������tz���������������xut7<HNJLMHC<5202777777����%-21+���������������������������)5BN[gt����t[N5$��������������������������������55BCDGBBA54455555555������ ���������������������������������� �����������1O[t��������h[O?6,,1��������������������������������������������������������������������������������;CHWahmjjnuqnaUH<;=;��������������������MOW[hmsih[OLMMMMMMMM;N[mtyuzt|ztg[QB88;��������������()6@BBB<6)#"((((((((��������������������ntz�������������tpnn����������������������������������������
#(/<@HIF<:/-#��� 
#/3<@DA</#
 ��f�\�_�f�O�R�^����׾����� � �	����Y�M���.�4�@�M�f�r���������������r�Y�;�.�����;�G�`�m�y�������������m�T�;���������������ѿ���������ݿĿ������������U�=�1������g�����������������������������������������(�/�,����������5��$�(�F�N�g�s�������������������s�Z�5F$FF F(F1F=FJFYFcFxF|F�F|FoFcFVFJF=F1F$�������������������
��#�,�0�.�(�#��
�������������� �!�&�!����������H�/����#�<�D�H�U�i�zÇÐÓÑÌ�z�U�HŹŵŭŭŭŭŹ��������������������žŹŹ��������������������àÜÝààìù������������������ùìàà�� ����������������������A�7�4�.�1�4�A�M�R�Z�[�Z�X�X�M�I�A�A�A�A�"�����"�#�/�0�;�E�G�;�/�"�"�"�"�"�"�;�1�5�A�G�T�m���������ѿݿ��ѿ��y�`�;���x�l�a�f�l�z���������������������������ݽѽнȽǽннݽ������������ݽݽݽ�ìçèìù��������ùìììììììììì�������(�5�N�P�Z�_�]�Z�M�A�5�(���;�.��	�������	�"�.�G�T�R�U�[�`�d�`�T�;����������������������"�(�+��������n�V�I�:�A�E�I�O�V�b�oǈǔǝǣǤǗǈ�{�n������������������������������������Ѽʼ¼ּܼ����!�.�:�F�E�:�/�����ּ�����	�
�����*�*�*�(� ��������۾ɾ��þ̾۾��	��&�4�4�.�1�,������H�E�<�:�7�<�H�U�a�b�d�a�a�U�H�H�H�H�H�H�׾Ѿ׾پ�����������׾׾׾׾׾׾׾��n�m�k�n�{ŇŔŕŔŋŇ�{�n�n�n�n�n�n�n�n�N�H�9�4�2�?�N�Z�g�s�������������s�g�Z�N���
���6�X�uƁƎƳƿƼƲƧƁ�h�C�*��0�(�%�/�0�=�I�V�b�f�o�r�s�o�k�b�V�I�=�0���s�l�c�h�s�����������������������������������������������������������������ìáàÙ××àììíù����������ù÷ìì�_�Z�S�S�G�K�S�_�l�x�x�~�����x�l�_�_�_�_�3�1�/�3�6�<�@�F�L�Y�e�s�����y�r�^�B�@�3�a�H�?�;�A�H�T�a�m�|��������������z�m�a�ֺκɺ������������ɺֺ����������ֹù����������ùϹѹܹԹϹùùùùùùù������������������*�C�O�T�N�A�*������߼�����{�v�t�s�w������������������������f�e�[�f�q�r�s�������r�f�f�f�f�f�f�f�f�:�-�!���-�:�F�S�_�f�l�x�{���x�l�_�F�:����ĿĽ�����������������
�������������������������
�������
���������غɺź������ɺԺֺ����ֺɺɺɺɺɺɺɺ�E�E�E�E�E�E�E�E�E�E�E�E�E�E�FF	E�E�E�E�E*E$E*E-E4E7E?ECEPE\EdEhEbE\EVEPECE?E7E* 3 > F R b 2 M Z  0 9 : g ? g A < v J  U A Y =  Z f @ : 5 7 ^  U  ; n 6 D _ B K < c G < g 6 6 z G ^  X  U  �  t  �     �  �    P  �  �  F  D  7  �  �  
  �  �  �  �  �  X  _  �  4  �    �  �  U  �  w  �  �  �    *  �  �  �  �    	  `  �  �  �  ;  �  i��7L�y�#�\)�ě����
��/��C��C���o�49X��hs��C���C���1���
��/���ͽ<j�aG��t��t��Y��@��,1��%��㽅��+�y�#�49X��w�t���o������+��;d�#�
�q���y�#�u�y�#�q��������T�� Ž�+��C����P��C�����	7L�$�B �!B ��B̜B	��B&��B5OBA�B(�B3B%��B�<B�tBV�B O]B ��B�"A�H�BT�BӱB��B��Bf�B/��B��B%B�sB-�B�JB�)B!<�Bj�B�B�vB/Bi�BL�B,'CB8#B"	�B�IB�VB��BT�B�UB�;B��B�!B
��B��B#��B� B�BӾB �:B�KB
EtB'FBHfBˋB?3B>�B%�B=EB1�BA�A���B �UB��A��B6FB�B�B1�B=�B/��BK�B>[B�dB.F�B��B	.B!?�B>4B�BAkB�IB��BB}B,f{B$NB"/uB?�B��BB�BB�BJhB?�B��B��B
?�B�+B#��B@�B�zAM�l@�ֆAf�AzA��JA��IA��C��^A���A
*A�R(A�KvA��`A�ѫA�&
A;g1A��ApC�@�SA+s�A�snA�<~A_r~A���B��A�S�A:�A�1�AY��A�/IAV+A��hA��'B �-B]LA�!�ArẢ@��_?�N�A�/@;�3>\�A��@�,@�s@�}�A��IA�	�@:��C�n2C��AP�@� .Ae�?A{�qA��QA�|�A�+IC��IA��yA
E�Aņ�A�~gA���ÂvA�z�A:��A���Awޚ@��UA+;�À�A�(A^��A���B<]A�A
�TA�s1AY�1A��:AV��A�`�A���B ��B��A��AsfA�}@��?�)�A���@6��>GoA���@�5�@��@�$A��A捏@8��C�y�C��-   H   B   &      P             =      =                        "                  $      &                     ?      I                           "                  /   (   C   )   -   #   M   '   #            +                     -               %            %      '            !   +      3                        #                           3         !   C      #            '                     +                           %      !                     #                                                Pc�O7O%�[O���P��Ow:[Oď�N��~OY��NC4O�=FN�x�M��O	5�M���Nw_�N��fP}EOr-�NՆNT�dOBeIO6��O�`MO���N�~O��^N]��O�ЯNY�5N�@BN��O&ڌO�@�OD��O�ސN?VTN�:�O��O2A�O@l�O2J=N3]N���O]&HNb�O72�N��N�F
N�OpUNО�  d    
  .  z  �  �  �  
�    !  O  �  A  A  �  �  q  �  �  T  �  �  |  �  �  �  �  R  ^  �  �  k  �  �  �  E  �      �  h  �  3    x  �  �  �  v  
�  
�u��h��t��ě��ě��49X�o�D���T���o��9X�D���e`B��o��C���t����㼴9X�����ͼ��ͼ����o��`B�+��`B��`B��h��P�+�+�+�D���u��㽃o����w�49X�@��H�9�D���Y���7L�aG��Y��aG���o�u��O߽�9X����t���������������root��������������������#)6;BFLKMPOB6*)# #[gt���������qg_XUSU[���
#0h�������{U<0��������� ����������>HUWnr�����zaUNHA:8>lnz������|zrnjghllll��������������������'06<>AA<90-%''''''''$/<HYa\a|�fUH</#������

������������������������wz�����������{zytqww��������������������FHLUadhkaULHFFFFFFFF�����		
	������������������
�������56BOZs}utg[OB;6.,*05

#//43/*#
)-24)!��������������������!*67CHKKIDC6*
����������������������������������{yxz7<HNJLMHC<5202777777����%-21+���������������������������&5BN[gt�����t[NB5+$&��������������������������������55BCDGBBA54455555555���������������������������������������������������������?O[t}��}xwsmh[OH@89?��������������������������������������������������������������������������������<?DHUagliiknrncaU<><��������������������OOZ[hhphf[ONOOOOOOOOGN[giihge][YNHCBGGGG��������������()6@BBB<6)#"((((((((��������������������stt{�����������ztsss����������������������������������������
#(/<@HIF<:/-#���
#+/:<?<:/# 



���x�y�����f�a�e������׾����𾾾����M�H�@�4�:�@�F�M�Y�f�h�q�r�x�t�r�f�Y�M�M�.�'�+�.�5�;�G�T�`�m�r�x�u�m�l�`�T�G�;�.�����������Ŀѿݿ����
����ѿĿ����������������Z�K�(��	��5�s������������������������������������� �	���"��	�������5��$�(�F�N�g�s�������������������s�Z�5F1F-F)F1F2F=FJFOFVFcFgFcFaFVFJF=F1F1F1F1�������������������
��#�)�,�*�$�#��
�������������� �!�&�!����������H�/����#�/�<�C�H�U�a�~ÆÊÈÃ�n�U�HŹŵŭŭŭŭŹ��������������������žŹŹ��������������������àÜÝààìù������������������ùìàà�� ����������������������A�:�4�/�2�4�A�M�V�V�M�G�A�A�A�A�A�A�A�A�"�����"�#�/�0�;�E�G�;�/�"�"�"�"�"�"�`�;�3�7�C�H�T�y�������Ŀѿݿ��ѿ����`�x�u�m�s�x�����������������������������x�ݽѽнȽǽннݽ������������ݽݽݽ�ìçèìù��������ùìììììììììì�������(�5�N�P�Z�_�]�Z�M�A�5�(���.�"��	�������	��"�.�;�<�D�F�B�D�;�.�����������������������$���	������V�I�A�G�I�J�V�o�{ǈǔǙǠǡǔǈ�{�o�b�V������������������������������������Ѽʼ¼ּܼ����!�.�:�F�E�:�/�����ּ�����	�
�����*�*�*�(� ��������վɾȾҾ����	��"�,�-�&�)�'�"������H�F�<�;�<�H�I�U�`�a�c�a�`�U�H�H�H�H�H�H�׾Ѿ׾پ�����������׾׾׾׾׾׾׾��n�m�k�n�{ŇŔŕŔŋŇ�{�n�n�n�n�n�n�n�n�g�[�N�G�F�N�W�Z�g�s�}���������������s�g�%������#�*�6�C�\�p�u�q�h�\�O�C�6�%�0�*�&�+�0�=�I�V�b�e�o�q�r�o�j�b�V�I�=�0�����~�z�����������������������������������������������������������������������ìáàÙ××àììíù����������ù÷ìì�_�Z�S�S�G�K�S�_�l�x�x�~�����x�l�_�_�_�_�3�1�/�3�6�<�@�F�L�Y�e�s�����y�r�^�B�@�3�a�T�H�@�;�B�H�T�a�m�z�������������z�m�a�ֺκɺ������������ɺֺ����������ֹùù��������ù͹ϹٹйϹùùùùùùù����������������������������������߼�����|�x�v�u�x������������������������f�e�[�f�q�r�s�������r�f�f�f�f�f�f�f�f�:�-�!���-�:�F�S�_�f�l�x�{���x�l�_�F�:���������������������������������������������������������
������
��������ɺź������ɺԺֺ����ֺɺɺɺɺɺɺɺ�E�E�E�E�E�E�E�E�E�E�E�E�E�E�FF	E�E�E�E�E7E-E2E7E9ECEGEPEYE\EcE_E\ETEPECE7E7E7E7 5 ) < Y X ) M 8  0 3 : g ? g ' < t 7  U A 7 9 ! Z f @ ? ) 7 ^ & #  > n 6 D _ ? K = / @ < g + > z G N  �  %  j  �  �  �  �  �  �  P    �  F  D  7  �  �    �  �  �  �  �  !    �  4  �  �  a  �  U  d  S  �  �  �    *  �  �  �  L  �  �  `  �      ;  �    A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  �    +  G  ^  d  ^  @    �  �  �  `  @    �  _  �  L  �    .  S  v  �  �  �  �          �  �  v  $  �  �  �  �  f  �  �  �  �  �  �  �  �  �      �  �  �  Q  �  {  	  �  �    !  *  .  *    	  �  �  �  l  7  �  �  �  Z    �  -  i  d  j  \  E  =  1    �  �  i    �  n    �  m  �  3   l  V  x  �  �  �  �  �  �  �  �  t  P  +    �  �  z  Z  ,  �  �  �  �  �  o  S  H  =  2  (  !        �  �  �  �  �  I    X  ~  �  �  �  �  �  �  s  R  "  �  �  A  �  5  �    �  
�  
�  
�  
�  
�  
�  
�  
k  
,  	�  	|  	  �  "  �  �  b  �  -   a        �  �  �  �  �  �  �  �  �  �  �  �  �  x  m  b  W  �  �        �  �  �  �  �      �  �  �  e    �  �  B  O  E  :  0  %      �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  A  )    �  �  �  �  �  �  �  �    |  �  �  �  �  �  z  t  A  ?  >  <  ;  9  8  6  5  4  3  2  0  /  -  +  )  '  %  #  �  �  �  �  �  �  �  �  �  x  ^  C  (    �  �  �  �  m  K  �  �  �  �  �  �    o  ^  L  8  !    �  �  �  |  1   �   �  f  n  `  N  7  .  *  "    �  �  �  j    �  �  �  w  8  T  i  �  �  �  �  �  �  �  |  ^  :    �  �  E  �  �  E  �  �  �  �  u  e  Y  Q  C  4      �  �  �  �  �  v  \  F  5  (  T  S  S  R  P  N  J  G  C  >  7        �  �  �  d  8  
  �  �  �  �  �  �  �  �  z  i  W  >    �  �  6  �  @  �    u  =  ^  q  w  {  �  �  �  �  �  m  O  (  �  �  �  X    �   �  u  z  v  p  l  g  a  Y  O  D  9  ,    �  �  �  u  d  t  �  �  �  �  �  �  �  �  |  l  X  <    �  �  �  G  �  6  a  a  �  �  �  �  s  b  V  U  J  0    �  �  �  �  a  8    �  �  �  �  �  �  �  �  �  ~  U  $  �  �  p  b    �  I  �  @    �  �  �  �  s  e  U  F  7  '      
    �  �  �           5  G  P  Q  E  ,    �  �  �  �  �  �  s  C  �  �  @  �  =  R  V  B  +    �  �  �  �  j  ?    �  �  l  *  �  �  W  �  �  �  �  �  �  �  �  t  a  I  +    �  �  �  l  =     �  �  �  �  }  l  [  K  6      �  �  �  �  �  w  ]  C  *    �    !  0  =  J  U  _  h  j  c  O  1     �  x  '  �  y  S    *        �  �  �  �  �  {  4  �  v    �    Y  d  �  �  �  �  �  �  �  �  ^  +  �  �  �  �  a  *  �  x  /  �  �  m  ^  q  �  �  �  �  �  �  �  �  �  �  @  �  {  �  0     �  E  >  7  0  )  "           �   �   �   �   �   �   �   �   �   �  �  �  �  �  �  �  �  |  d  J  /    �  �  5  �  )  �  �  Y    {  w  v  w  z  z  t  h  R  2  �  �  v  +  �  m  �  z   �    
  �  �  �  �    U  ,    �  �  �  d  0  	  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  w  Y  9    �  �  �  �  m  h  Z  L  9  #  
  �  �  �  �  �  w  W  1    �  �  �  e  8  @  _  }  �  �  �  �  �  n  T  9    �  �  �  �  d  .  �  �  �  �  �  �  �  �  �      ,  /    �  �  s  %  �  z  i  k  �  	  �  �  �  �  P    �  �  C  �  �  U    �  u  �  m  �  x  m  \  H  5     	  �  �  �  �  X  "  �  �  *  �  :  �  3  �  �  �  �  �  �  }  s  b  N  C  .    �  �  �  �  �  �  �  q    �  �  �  �  �  �  �  �  �  �  b  ?    �  �  �  U    }  �  �  �  �  �  �  u  f  T  =  !    �  �  �  `  (     �  v  i  \  N  A  3  $      �  �  �  �  �  �  �  �  �  �  �  
�  
`  
-  	�  	�  	f  	   �  �  l  J    �  c  �  I  �  �    *  	�  	�  	�  	�  	�  	�  	�  	�  	�  	H  	   �  _    �  7  �  E  �  �