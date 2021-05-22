CDF       
      obs    5   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?��\(�      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       N.�   max       P�.�      �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       �ix�   max       >J      �  T   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>���R   max       @F��Q�     H   (   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�
=p��    max       @vnfffff     H  (p   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @,         max       @Q�           l  0�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @�u�          �  1$   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �T��   max       >fff      �  1�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�ś   max       B4M      �  2�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�s�   max       B4!�      �  3�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >�?�   max       C�O�      �  4t   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >���   max       C�`�      �  5H   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          �      �  6   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          ;      �  6�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          7      �  7�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       N.�   max       P�      �  8�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��_o�   max       ?�X�e,      �  9l   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       �ix�   max       >�      �  :@   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�
=p��   max       @F��Q�     H  ;   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�
=p��    max       @vnfffff     H  C\   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @         max       @Q�           l  K�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @���          �  L   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         E�   max         E�      �  L�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��t�j~�   max       ?�n��O�     �  M�                     	      	                                    w   W   7            �   '         �      5      �      3         8               !         	   w      K      QN�"�NE��NH@MO1z�O�@ N|p�N�#�NF�O�VO�=�OnOٴOϹ�NY��N�JN�&N��0N.�P�O�ohP�.�P��(OܜN�ʦO -O�$OP]>O2?eN�0xO4�'PN�7N���P��NmئP$3YN&iiO\N�O�rN�6#O�E�N��!NZlO���O�ŋO�diN��>N�N�Nz pO��fNkHmO��O=�O~�Žixսixռ����u�D���49X�t��t�;o;ě�;�`B<t�<t�<D��<T��<T��<�o<�o<���<��
<��
<��
<�1<�9X<�h<�h<�=C�=t�=�P=��=#�
=49X=8Q�=D��=L��=L��=Y�=Y�=]/=u=�7L=�C�=��=���=���=���=��-=��=�{=�"�=�"�>J~����������������������������������������������������#,/2<@DFF?</#vy���������������{v�������������������,*./<HKOSH</,,,,,,,,bhlt�����tkhbbbbbbbb�������
����!!#0<IbfebUI?0'!#./0<DHSTNH</%%#
"/;?HPHFB;3/"���������������������������������������������������������������������������),575-)������������������������ZZamz�����������zmbZ�������������������������&+2DH?)���������)@OjkdXNB5��BGV_hru�������h[OHDB������������������������������������������������������������%!&7N[g��������g[B7%������������������������������������������������������������������������������������������������������������
���������

�������������������#
�����������������������������
#)/233/%#
�����)01/)	�����()151)"����������������������������������������
#-(#
���	)BLPQNB6,�����)6BA<6)*)��������&+++(%"�('%")0569;:951,)((((��������������������������������������������������
����������������������������������������������������

������������n�{ŁŇŏŊŊŇ�{�s�n�m�i�l�n�n�n�n�n�n�����ĽǽнܽнĽ���������������������������*���	������������������(�5�7�A�S�Z�d�Z�L�A�5�(����
���$�(��(�5�A�a�j�g�f�h�Z�N�A�.�(������E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�EEEEE%E!EEED�D�EEEEEEEEE�G�O�T�Z�W�T�G�;�8�6�;�<�G�G�G�G�G�G�G�G�����������������������������������������f�r������������������������t�`�Y�R�Y�f�����������������������������T�a�m�o�t�x�y�z���z�r�m�a�Z�T�Q�I�K�T�T�H�a�m�x���������z�a�H�;�/�����"�/�H�	���������������	�	�	�	�	�	�	�	����������������������������������������������������üùòøù�����������������žf�g�j�k�h�f�\�Z�W�M�J�J�M�R�Z�f�f�f�f�fàåìììàßÓÏÑÓÓàààààààà���������������������������y�p�s�q�s�����tāčĚĤĪĵĹķĲĦĚčā�u�d�c�e�n�t�
�<�U�xŎŝŠśŇ�b�I�#��������ĸ�����
������;�]�d�h�f�T�;��	����������������'�@�Y�r�w�f�Y�@�4�����ܻӻ������'�4�A�M�Z�f�h�i�f�_�Z�M�A�7�4�.�/�4�4�4�4���#�*�3�6�8�6�2�*��������	��������'�4�@�M�G�'�����޻ܻջܻݻ����)�B�]�c�f�]�O�6�����������������ìù������ùìàÚÓÇ�{�w�x�zÆÇÓàì�6�?�B�E�O�[�]�]�_�[�U�O�B�>�<�6�)� �)�6��!�5�A�N�Z�g�s�y�s�g�T�N�A�3� �����Üåè÷þ��üìÇ�U�H�J�M�U�\�a�f��}Ü���������ĿпѿտѿĿ��������������������[�g¦¿������¿¦�t�N�5�)�"��-�D�[�H�N�O�H�G�;�4�/�*�-�/�/�;�=�H�H�H�H�H�H���!�;�M�W�X�P�-������ߺɺ������ֺ�²µ¿��������¿²±°­²²²²²²²²�`�m�y�����������~�y�m�`�X�G�E�>�@�G�T�`��(�5�6�:�5�*������ӿԿݿ��������������Ŀ˿Ŀ��������������������������������(�4�E�K�H�A�3�(������������Ϲܹ�������ܹϹƹù¹ùǹϹϹϹ�ǡǪǭǯǭǬǡǔǐǐǔǗǡǡǡǡǡǡǡǡ�׾�����	�����վʾ����������������׺e�r�~�������������~�r�g�Y�L�@�.�6�<�L�eƳ����������������������ƧƚƎƇƎƒƞƳ�O�\�h�uƀ�u�l�h�\�O�C�6�.�6�C�M�O�O�O�O�����������������������������������������/�<�H�U�W�W�U�H�<�/�(�-�/�/�/�/�/�/�/�/D{D�D�D�D�D�D�D�D�D�D�D�D�D�D�D{DmDjDlD{������������ �����������������������޼���'�4�8�A�A�9�.�����ܻӻֻ���ûλлѻʻĻ��������������������������E�E�E�E�E�E�E�E�E�EtEoEnEqEuEzE�E�E�E�E� L f f 5 c F - :  0 E c # Y G � { O  9 J 4 L  ( E  T I u ; c g i $ m  ' K $ N J ` / @ � * 5  M E Y '    �  �  q  �  �  �  �  j  Q  w  "  `  �  �    �  #  F  }  �  �  �  -  �    *  x  �  �    t  �  9  �  �  s  �  u  �  Y  �  ~    �  =  �  �  {    �  �  �  �T���T�����
%   <T��%   ��o�ě�<#�
=C�<�/<�9X<���<u<�h<���<�j<�9X=D��=D��>O�=�"�=���=,1=@�=ix�>fff=�t�=H�9=@�>J��=8Q�=ě�=L��>F��=ix�=��=��
=m�h=�G�=�hs=��P=��=��=�"�=��T=� �=� �>L��=�j>9X>C�>T��B
�B�&Bd�B5nB��BYB�1B<�B�GB&L`B�A�śB}�B4MB�B{�BT�B�B ��B��B��ByBj	B��B�^B!�B	^B"I�B�4B�B�qB�B~wB��B"]�BSB8B��B|qB"/MB��BD�BpBe=B�iB(LB- �Bw�B��B!�Bq�BE�BZxB
ɆB<�BAfBAAB;-B=sB�^B?!B|QB&>�BŝA�s�B��B4!�B'B�BY�B��B �kBǔB<$B�cBC�B�dB��B!�6B�/B"7B>�B�B�zB@�B�B��B"@	B�bB?�B�$B��B"?�B��B@XB #B��B�,Ba�B-�BCB�\B*B�0B�B��A�_A&*QA�26A�Y�A��C�O�C�d�AeEA�W@�?A�|�A���A��AX�A���A��FA?];A�@A�bAއ�A�8vA�e@���A<�[A�Px@�@-AՐ�A��RA���A�A�`�Aw9A��eA��~@\��A�(�Aj��A�3�At��A4#>�?�B��AQ��?��RB��B��A"��A���C�ԅA�/@�Ai@���C��A�SA'fA���A�A�}�C�`�C�h;Ae�A��@�'�A҄xA��A�K�AX
�AϥA΂�A>�A�~�A�eA�g�A� A�w@��WA=�A��@�V�A�rA��@Aؓ�A��oA�[Aw�A�!�A��@[�A�LAj��A�|�As^A4�>���B��AP�?�#EB�~B\�A"��AÃ=C��A���@��@��C�h                     	      	                                    w   X   7            �   '         �      5      �      4         9               !         
   w      K      R               %                        !                  #      9   ;   )            -            /      -      -                        %   #                                                                  !                  %      5   7               !                  +                              #   #                           N@t&NE��NH@MO1z�N�d�N|p�N�#�NF�N�POo5'Nz21N�ԃOϹ�NY��N�F�N�&Nr|�N.�P�O�ohP�P{!uN���N�ʦO -Oy��OίzO ��N>y�O��O`��N���P �NmئO��XN&iiOE��O�rN�6#O�N}N��!NZlO��ZO�ŋOd�N��>N�N�Nz pO1�NkHmOp/�Ov�Op:�  �  &  �  x  �    b  N  �  )  *  $  }  �  �  �  �  �  �  �    n  e      �  �  ,  �  c  �  j  {  k  -  �  	�    E  >  �  �  7  �  �  �  �  *  �  �    	i  !�e`B�ixռ����u%   �49X�t��t�;�o<D��<T��<49X<t�<D��<�C�<T��<�C�<�o<��
<��
=C�<�=P�`<�9X<�h<�=��='�='�=�w=�F=#�
=D��=8Q�=�"�=L��=Y�=Y�=Y�=y�#=u=�7L=�\)=��=��-=���=���=��->%=�{=��#=�/>�������������������������������������������������������������#,/2<@DFF?</#���������������������������������������,*./<HKOSH</,,,,,,,,bhlt�����tkhbbbbbbbb������������ 0<IU\`^YUOI<0.'$,-/0<HLPHH</,,,,,,,,"/;C@;//"��������������������������������������������������������������������������

)+565,)





��������������������[[amz����������zmda[��������������������������#)?C:)������)BIde`TB5.���OLOO[[`hrrnhc[VOOOOO������������������������������������������������������������?979?N[gt������tg[I?���������������������������������������������������������������������������������������������������������������	
����������

������������������������������������������������������
#(/122/+#
������)01/)	�����()151)"����������������������������������������
#-(#
 ��)BJOQOKB6. ����)6BA<6)*)�������	%)++*&#��('%")0569;:951,)((((������������������������������������������������������������������������������������������������������������

��������{�|ŇŋňŇ�{�q�n�l�n�w�{�{�{�{�{�{�{�{�����ĽǽнܽнĽ���������������������������*���	������������������(�5�7�A�S�Z�d�Z�L�A�5�(����
���$�(�N�N�Z�]�^�\�Z�Q�N�I�A�6�5�-�,�5�A�E�N�NE�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�EEEEE%E!EEED�D�EEEEEEEEE�G�O�T�Z�W�T�G�;�8�6�;�<�G�G�G�G�G�G�G�G���������������������������������������˼����������������������r�h�_�X�Y�f�r������ �������������������������������T�a�m�m�s�u�u�m�a�\�T�T�L�M�T�T�T�T�T�T�H�a�m�x���������z�a�H�;�/�����"�/�H�	���������������	�	�	�	�	�	�	�	����������������������������������������������������üùòøù�����������������žZ�f�h�j�g�f�[�Z�Y�M�K�L�M�S�Z�Z�Z�Z�Z�ZàåìììàßÓÏÑÓÓàààààààà�������������������������{�q�s�r�u�������tāčĚĤĪĵĹķĲĦĚčā�u�d�c�e�n�t���
�#�I�fŃŖŗŎ�{�U�#������������������;�W�^�c�_�H�;��	���������������������'�4�4�@�A�@�@�4�'�����������4�A�M�Z�f�h�i�f�_�Z�M�A�7�4�.�/�4�4�4�4���#�*�3�6�8�6�2�*��������	��������'�4�@�K�E�'��������߻ܻ߻�����)�6�H�Q�S�N�B�6��������������Óàìù��������ùïìàÓÇ�~�|ÇÐÓÓ�)�6�B�O�O�X�O�B�>�6�)�$�)�)�)�)�)�)�)�)��(�5�A�N�Z�g�s�u�s�g�d�Q�N�A�:�5�$���n�zÇÓßàèêçàÙÓÇ�z�n�d�a�c�m�n���������ĿпѿտѿĿ��������������������g¦¿������¿¦�t�[�B�5�)�!�0�B�[�g�H�N�O�H�G�;�4�/�*�-�/�/�;�=�H�H�H�H�H�H������3�;�=�:�/�!�������ٺӺغ��²µ¿��������¿²±°­²²²²²²²²�m�y�������������|�y�m�`�[�T�G�A�G�T�b�m��(�5�6�:�5�*������ӿԿݿ��������������Ŀ˿Ŀ��������������������������������(�4�@�G�D�A�4�-�(�����������Ϲܹ�������ܹϹƹù¹ùǹϹϹϹ�ǡǪǭǯǭǬǡǔǐǐǔǗǡǡǡǡǡǡǡǡ�ʾ׾���	����Ӿʾ������������������ʺe�r�~�������������~�r�g�Y�L�@�.�6�<�L�e��������������������������ƧƚƎƔơƳ���O�\�h�uƀ�u�l�h�\�O�C�6�.�6�C�M�O�O�O�O�����������������������������������������/�<�H�U�W�W�U�H�<�/�(�-�/�/�/�/�/�/�/�/D�D�D�D�D�D�D�D�D�D�D�D�D�D�D}D�D�D�D�D������������� �����������������������޼��'�.�4�9�:�2�(��������������ûǻлѻɻû���������������������������E�E�E�E�E�E�E�E�E�E�E�E�E�EuEpEoEsEuE}E� O f f 5 2 F - :  0 1 R # Y 7 � S O  9 F 5 !  ( H  V v ]  c d i   m  ' K  N J ] / 9 � * 5 ! M 3 Y )    g  �  q  �    �  �  j  �  �  �     �  �  �  �  �  F  c  �  ^    �  �      �    �  g  �  �  �  �  #  s  �  u  �  �  �  ~  �  �    �  �  {    �  �  �  �  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  �  �  �  �  �  �  �  �  �  �  �  �  �  �  u  g  T  @  ,    &        �  �  �  �  �  �  �          �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  p  ^  K  8  %           x  w  w  u  p  j  b  X  M  B  7  +      �  �  �  t  ?    G  O  P  L  G  C  <  a  ~  �  �  �  _  =    �  �  �  �  �      �  �  �  �  �  �  �  h  L  0      !    �  �  �  �  b  T  E  6  '      �  �  �  �  �  �  �  g  K  /    �  �  N  L  J  I  G  E  D  <  1  '         �   �   �   �   �   �   x  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  q  Y  A  �      '  )  #      �  �  �  �  _  #  �  �  R  <  
  �  �  �  �  �  �  �  (  "      �  �  �  �  `  (  �  �  1  �  �  �    #           �  �  �  �  �  N    �  �  <  �  m  }  x  p  h  `  X  Q  R  Q  O  K  C  6  %    �  �  �  e  0  �  �  �  �  �  �    z  t  n  h  b  \  X  Y  Y  Z  [  \  \  �  �  �  �  �  �  �  �  �  �  �  x  V  3    �  �  �  V  �  �  �  �  �    7  ;  9  3  0  *  $  !          8  ~  �    L  {  �  �  �  �  x  g  K  -    �  �  �  f  0  �  �  n  �  �  �  �  �  �  �  �  ~  n  Q  '  �  �  �  �  �  P    �  �  �  �    f  I  +  	  �  �  �  _  $  �  �  t  %  �  �  X  �  �  �  z  ]  4    �  �  f  ,  �  �  o     �  W  �  E  �  
�  
�      
�  
�  
V  
  	�  	K  �  �  o  )  �  �    C  �    5  \  n  i  Q  +    �  �  �  �  �  ^    �    \  �  �  Y  �      3  @  ?  C  �    G  a  ]  B    �  m  �  �  A  �        �  �  �  �  �  �  ]  .  �  �  y  &  �  Y  �  D  �      �  �  �  �  �  �  �  �  ^  3  �  �  |  3  �  �  >  �  �  �  �  �  �  �  x  n  a  I  &        �  �  �  �  m     �  $  �  �  .  �  �  �  �  �  b  �  b  �  �  |  	    �  �  �  �    "  )      �  �  �  Q    �  d  �  6  �  �    9  �  �  �    n  o  �  �  �  ~  �  �  �    +  F  _  v  �  �  b  I  5  O  _  K  5    �  �  �  �  �  x  Y  9    �  �  �  	^  �  �    �  w  �  h  �  �  �  �  m  �  �  �  Z  
�  k  �  j  `  V  L  @  +    �  �  �  �  �  �  �  �  �  �  �  �  �  s  g  r  Z  ;    �  �  a  N  }  �  q  H    �  w    �  �  k  p  t  y  |  y  u  r  o  l  i  g  d  `  ]  Z  X  V  U  S  *  �  |    k  �  �  %  ,    �  p  �  <  u  
�  	�      �  �  �  �  �  �  �  �  �  x  X  8       �  �  �  �  �  �  �  	�  	�  	�  	�  	�  	�  	l  	?  	  �  }  $  �  ]  �  J  �  �  &  V        �  �  �  �  X  (  �  �  �  t  R  .    �  !  �  :  E  .      �  �  �  �  �  �  �  �  �  �  ~  r  f  Y  M  @  �  )  :  =  :  4  +      �  �  �  B  �  �  *  �  �  �  �  �  �  n  J  >  G  2    �  �  �  �  g  =    �  �  �  �  �  �  �  �  �  �  �  �  �  {  Z  6    �  �  �  w  Y  9     �  .  6  /    �  �  �  �  �  X  #  �  �  x  2  �  �  T    �  �  ;  3  T  O  B  $  �  �  �  x  z  a  /  �  �  0  �    �  �  �  �  �  �  }  Q     �  �  �  L    �  w    r  �  �   �  �  �  �  �  �  �    i  S  =  F  n  �  �  �  �  �  w  i  Z  �  �  ~  o  _  N  9  %    �  �  �  �  �  �  g  G  %  �  �  *  
  �  �  �  �  �  �  �  �  k  Z  O  D  7  *      �  �  m    �    m  �  �  �  �  �  a  �  ~  �  �  �  �  
  �  �  �  }  w  p  g  ^  U  J  @  ,      �  �  �  �  �  �  �  �  �  �  �  �    �  �  �  �  Z    
�  
a  	�  	E  �  �  �  �  �  	e  	d  	C  	  �  �  O    �  q  *  �  �  C  �  |    �  m  /  �       �  �  o  L  !  �  �  V  �  l  �  %  
#  �  �  �  �