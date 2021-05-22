CDF       
      obs    <   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?�            �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M��]   max       P��      �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       �C�   max       >I�      �  �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>Y�����   max       @E�z�G�     	`   |   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?��\)    max       @vo��Q�     	`  )�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @-         max       @Q            x  3<   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�?        max       @�x�          �  3�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �e`B   max       >j~�      �  4�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A��   max       B,h6      �  5�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A��   max       B,��      �  6�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       ?��   max       C�i'      �  7t   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       ?��   max       C�g�      �  8d   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          �      �  9T   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          ?      �  :D   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          #      �  ;4   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M��]   max       O��V      �  <$   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�Q�_p   max       ?��8�YJ�      �  =   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ����   max       >I�      �  >   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>Y�����   max       @E�z�G�     	`  >�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?��\)    max       @vo��Q�     	`  HT   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @!         max       @Q            x  Q�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�?        max       @���          �  R,   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         @<   max         @<      �  S   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?u�s�g�   max       ?���a@O     �  T   )      �            @         	            !   e   8   %   "             +                  ^   
   #   %      
      $   
   #               
                                 G               
      O�:�N���P{��N��iN6�+NP\PF׸O+��N�)N9H�N�0N!��N��O%P��P��Oϥ-O�VONS�8N[�:O��sO��Nl��O1�{M��]N�N� $Oȏ�N��sO���O�� O��N��Nc�-O��9NO+.�O]SZO`ިN5]}O�e(N�&�O��N�>hOFaO-�cN;�]O"g�NH̴N��>OkCTN��O#��N���N��_N|��ND'�N[T�N�Y�N�5	�C�������1���
�u�ě��ě�;D��;D��;�o;��
<t�<t�<49X<49X<49X<D��<e`B<e`B<u<u<��
<�9X<ě�<���<���<�`B<�h<�h<�h<��<��<��=o=+=+=C�=#�
=#�
='�=49X=L��=L��=P�`=ix�=q��=q��=}�=�\)=�hs=��=��-=� �=�E�=��>o>o>$�>$�>I�AL[gt�������tg[PJFDAommot�������|toooooo)5Nt��������gNB.#���������������������$&����������������������������!)BO[h���~��hVB6)lflmsxz|����������zl)*0)!��������������������ost�����}toooooooooo����������������������������������������������������������>=ANg�����������tNB>����
%<GHSHC<6/
�������  "!����������	����LNN[gttytg[NLLLLLLLL��������������#?LSUSLG20#�����������������������JNP[gijig[ONJJJJJJJJ��������������������?;@BCNPRNB??????????������������������������������������������
 /=FJHC</

#$&$#
JUWahtz���������znaJ���)7==<7+ ��4577=BN[cgijge[NB544����������������������������������������-.4<HUanz����zaUH<5-hjnpt�����������zunh#0<HMTURHA</+#~yxz~��������������~������������������������������������������������
&'#��������������������������(#)246BIOX[_[ZOB96)(������������������������,,+'"!�����������������@ABOO[gc[ONB@@@@@@@@�����������������������

�����������������������`^amz��������zma]\]`���������������������������������������������������������ymfddimz~�����zyyyyy<@><1/,# !#$/<<<<<<<���������������������������������������������������������������������������������
�����
��������������������������
�n�zÇÓßÞÓÉÇÃ�z�o�n�j�n�n�n�n�n�n��)�B�L�R�S�N�B�6��������åéò��������������������������������������������ؼr�v�������r�h�f�d�f�j�r�r�r�r�r�r�r�r�.�;�G�I�N�G�>�;�9�.�-�,�.�.�.�.�.�.�.�.�ʾ�����ܾ�����Z�M�>�5�A�U�k�������
��#�%�0�<�F�I�O�I�<�8�0�'�#�����
�T�U�^�^�T�T�H�B�;�:�6�;�H�J�T�T�T�T�T�T�/�<�<�?�>�<�4�/�%�%�*�.�/�/�/�/�/�/�/�/�Y�e�k�p�e�Y�L�L�L�N�Y�Y�Y�Y�Y�Y�Y�Y�Y�Y�6�:�B�B�H�L�B�B�?�6�)�$�)�0�6�6�6�6�6�6�/�1�<�H�U�H�?�<�:�/�#� �#�-�/�/�/�/�/�/�h�tāčďĚĦħĨĦĝĚčā�v�t�g�a�g�h��5�Z�x�����������g�C�(����	��������5�[�k�g�[�?�5�0�)�&�������������.�G�`�y�������������`�T�G�.����
��.�H�U�a�n�zÇËÊÇ��z�n�a�T�H�?�6�7�<�H�H�T�U�V�Z�T�T�H�?�B�B�E�H�H�H�H�H�H�H�H����������������ùùìëìôù���������ż�����������������r�f�M�H�@�H�Y�f������Z�f�s�{�{�w�w��s�f�X�M�4�*�&�)�/�4�@�Z�� �$�*�&�$����������������ûͻλǻû�����������������������������ƎƚƧƩƧƝƚƎƅƊƎƎƎƎƎƎƎƎƎƎ�#�/�0�<�B�<�/�#� � �#�#�#�#�#�#�#�#�#�#���������������������������������������Ҿʾ׾����"�+�;�.�#�	���׾ʾ������ƾʿ`�m�y�|�{�y�n�m�`�T�Q�G�<�G�T�Y�`�`�`�`��	���(�N�[�a�^�V�N�A�(���� ������#�<�I�Q�R�H�B�0�
��������������������#�I�L�U�b�n�r�{�y�n�g�b�U�L�I�B�>�A�G�I�I���������������������s�o�f�[�f�s�{���zÇÍÇÂ�z�w�n�a�^�a�i�n�r�z�z�z�z�z�z������������	������������������������������
�����
�����������������EEEE"E*E/E.E*EEED�D�D�D�D�D�EEE���������ĽнԽڽڽн��������������|����������(�4�>�C�A�4�(�����������E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E���"�/�;�H�F�<�0� ������������������	������������������������������������������������������кɺĺ��������������������������������������������������������������������û̻һݻܻлû������������~��������"�/�;�H�T�]�V�T�H�;�/�"���	��	����������������������������"�)�/�7�;�A�D�?�;�/�"� ��	���	��ǡǥǭǬǧǡǙǔǔǔǗǚǡǡǡǡǡǡǡǡ�� �!�,�-�#�!���������������������������������ŹŭŦŚŖřŭŹ������������
�������������������������D{D�D�D�D�D�D�D�D�D�D�D�D�D}D{DtDrDsD{D{�����������������������������������������z�ÇÓàìôñìãàÓÐÇ�z�z�z�z�z�z�6�*��������*�5�6�;�6�6�6�6�6�6�6�@�M�Y�f�k�g�f�Y�M�L�@�4�@�@�@�@�@�@�@�@²»¿����¿²¦¦¯²²²²²²²²�t�t�g�e�[�Y�[�g�i�t�t�t�t�����#�!����������������� P 7 ( $ f 0 M < 5 w / z c 4 3 # U % n v 6 9 ' @ T D N 4 I W I # K } L � G 5 G 0 ^ 5 W _ P J u R � ' 1 5 ) b 4  X 1 P    q  �  �  �  x  d  �  o  �  �  &  �  G  ]  �  K    �  �  �    �  v  �    2  �  �  �  �  B  O    �  W  �  �  �  �  J  �  �  N  �  �  �  �  t  �  �  �  �  f  �  �  �  i  m  �  �;ě��e`B>j~��D����`B�o=ix�<�C�<o<D��<�o<�1<e`B=0 �=�`B=�7L=H�9=@�<�C�<�`B=@�=�%<�/=P�`<�/=�w=�w=��#=��=�%=�7L=H�9='�=@�=�C�=0 �=�C�=�C�=�7L=T��=�O�=q��=�%=m�h=�-=�1=�o=��=��w=�{=Ƨ�=�^5>�w=��=�>\)>t�>bN>�>!��B	8�B
 �B��B0B8�BB��B )�B�B�B��B�B"9|B��B
�mB��B��B�qB��B|\B%(AB�uBąB!ӑB�XB$B�OB�B;WBe�B`BO�B��B!fBfTB�Bd'Bv�B!_�B��B޵Bg%B.sB,h6BC:A��B�"B˫BD1B��A�:BBo�B�DA�BD�B� B�>B?�BRyB	>.B
7.B��B0�BT�B�B��B 12B�VB @�B��B��B"@�B��B
��B>]B��B�B	9�B��B%��B�BB�B!��B'B?�B�UB��B=B�qB>mBz�B:�B�fB@$BAjB�B9�B!�|BSEBMeBG�BC�B,��B@hA��B�B�TB5�B��A�kB:�B��B<UA�{�B@�B�QB��B?�BT	A�&MA�A�A�~�A䤆@��Ac.�AJ�ZA��yA�ϛA�Y?��A��nA�A�AݼkA��A���Ac�AƋ�A���AΒ�@�W�A>4�B	0�@��fB'A�k�A�@AW��AirA���A�KA��@AE�PA��A�
OA�)\C�l�A"�A3��C�i'A�Q�A�$�@��A��@�k�A��@�~�A���B�@fS�A�1�AҎ�C��1A���AʯQA�8�@���A�a�A�A2��A�wAș�AӅzA�s�@��sAcҀAKGgA��A�}�A�i�?��A�s�A�s�AݐA���A�ѝAc�Aƒ:A�UÁx@��A=)�B	<�@�B�CA�A��AW~Ah�A�rA�\$A�`iAE�AȈA�xAA�xfC�hA#��A4��C�g�A���A�~�@$&�A @��~A���@� �A�c�B��@dA�uAү�C��lA�}KAʅ�A�h/@�EA��!A��bA2��   )      �   	   	      @         	            !   f   9   &   "         !   ,                  ^   
   #   %            $      #               
                     	            G                              1            3                        ?   %   '            '   !                           #                                                                                                                                                      #                                                                                                                     OyN���O��N��iN6�+NP\O���O+��N�)N9H�N�0N!��N��N�8�O��OtzOE'�Om�NS�8N[�:O��VO���Nl��O'�+M��]N�N7��Oo�&N��sO��O��+N��N��N��N]NNO+.�O6�O`ިN5]}O�-�N���O��N�>hO�O-�cN;�]O"g�NH̴N��>OkCTN���O\N���N��_N|��ND'�N[T�N��=N�5	  d  �  �    6  	  �  d  +  O  ;  b    R  S  =  �  	  �    �  �      F  7  �  F  �  '  �  .  U       n  �  �  �  �  y  �  �  N    �  '  �  �  �  i  �  R  �    �  U  8  9  1��9X����=�񪼣�
�u�ě�<�9X;D��;D��;�o;��
<t�<t�<�C�=�o=C�<���<�o<e`B<u<���<�1<�9X<���<���<���<�=aG�<�h<��=#�
=+<��=�P=Y�=+=C�=@�=#�
='�=8Q�=T��=L��=P�`=y�#=q��=q��=}�=�\)=�hs=��=��w=�v�=�E�=��>o>o>$�>+>I�LNN[gt}����wtg[TONLLommot�������|toooooo448BN[gtyzytmg[NB<64���������������������$&����������������������������557=BO[hmqtsoh[OB=85lflmsxz|����������zl)*0)!��������������������ost�����}toooooooooo������������������������������������������������������������fbekt������������tpf������
!)/0-%#
�����������������������LNN[gttytg[NLLLLLLLL���������������#0<FMPNIC0#����������������������JNP[gijig[ONJJJJJJJJ��������������������?;@BCNPRNB??????????����������������������������������������#/4;>A></#
#$&$#
V[diuz����������znaV���),58:7/)
��:9:@BN[bghigb[NB::::����������������������������������������98<HHIUaba[UH<999999hjnpt�����������zunh#0<HMTURHA</+#~}����������������~������������������������������������������������
$!��������������������������(#)246BIOX[_[ZOB96)(������������������������"(% ����������������@ABOO[gc[ONB@@@@@@@@�����������������������

�����������������������`^amz��������zma]\]`������������������������������ ���������������������������ymfddimz~�����zyyyyy<@><1/,# !#$/<<<<<<<�����������������������������������������������������������������������������������
�
����
��������������������������n�zÇÓßÞÓÉÇÃ�z�o�n�j�n�n�n�n�n�n����*�/�0�-� ������������������������������������������������������������ؼr�v�������r�h�f�d�f�j�r�r�r�r�r�r�r�r�.�;�G�I�N�G�>�;�9�.�-�,�.�.�.�.�.�.�.�.�������ʾҾ־Ͼ�����������s�r�v���������
��#�%�0�<�F�I�O�I�<�8�0�'�#�����
�T�U�^�^�T�T�H�B�;�:�6�;�H�J�T�T�T�T�T�T�/�<�<�?�>�<�4�/�%�%�*�.�/�/�/�/�/�/�/�/�Y�e�k�p�e�Y�L�L�L�N�Y�Y�Y�Y�Y�Y�Y�Y�Y�Y�6�:�B�B�H�L�B�B�?�6�)�$�)�0�6�6�6�6�6�6�/�1�<�H�U�H�?�<�:�/�#� �#�-�/�/�/�/�/�/�tāăčĚĢĢĚēčā��t�l�h�e�h�q�t�t�5�N�Z�g�s�|���{�q�g�Z�N�A�5�*�$�$�&�(�5���5�B�N�S�X�N�B�5�)������������.�;�G�T�`�j�m�p�m�e�`�T�G�.�"����"�.�H�U�a�zÃÇÊÉÇ�}�z�n�a�U�H�A�8�<�?�H�H�T�U�V�Z�T�T�H�?�B�B�E�H�H�H�H�H�H�H�H����������������ùùìëìôù���������ż���������������������v�f�O�F�O�Y�f����4�A�Z�f�s�{�{�v�w�~�s�f�Z�M�4�*�'�)�/�4�� �$�*�&�$����������������ûȻͻƻû�����������������������������ƎƚƧƩƧƝƚƎƅƊƎƎƎƎƎƎƎƎƎƎ�#�/�0�<�B�<�/�#� � �#�#�#�#�#�#�#�#�#�#���������������������������������������Ҿ����	��"�%�!��	�����׾ҾɾȾʾ׾�`�m�y�|�{�y�n�m�`�T�Q�G�<�G�T�Y�`�`�`�`����(�N�Y�_�\�T�N�A�5�(����� ����#�0�<�C�G�<�3�0��
�����������������
�#�I�U�b�n�o�x�w�n�d�b�U�O�I�D�@�D�I�I�I�I���������������������s�o�f�[�f�s�{���zÇÊÇ��z�p�n�a�a�a�k�n�u�z�z�z�z�z�z���������� �������������������������������������
�����
�����������������EEEE"E*E/E.E*EEED�D�D�D�D�D�EEE���������ĽνнĽý���������������������������(�4�>�C�A�4�(�����������E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E���"�/�;�E�D�;�/�"�����������������	������������������������������������������������������кɺĺ����������������������������������������������������������������������ûȻλлԻлû�������������������"�/�;�H�T�]�V�T�H�;�/�"���	��	����������������������������"�)�/�7�;�A�D�?�;�/�"� ��	���	��ǡǥǭǬǧǡǙǔǔǔǗǚǡǡǡǡǡǡǡǡ�� �!�,�-�#�!���������������������������������ŹŭŦŚŖřŭŹ�������������������������������������D{D�D�D�D�D�D�D�D�D�D�D�D�D�D{DwDuDwD{D{�����������������������������������������z�ÇÓàìôñìãàÓÐÇ�z�z�z�z�z�z�6�*��������*�5�6�;�6�6�6�6�6�6�6�@�M�Y�f�k�g�f�Y�M�L�@�4�@�@�@�@�@�@�@�@²»¿����¿²¦¦¯²²²²²²²²�t�y�t�g�f�[�Y�[�g�j�t�t�t�t�����#�!�����������������   7  $ f 0 / < 5 w / z c 0  * 6 $ n v 1 9 ' ? T D 3 2 I O K  K ~ S � G 4 G 0 W + W _ @ J u R � ' 1 3 ' b 4  X 1 N    F  �  0  �  x  d    o  �  �  &  �  G  �  b  �  �  �  �  �  �  y  v  s    2  W  �  �  d  �       o  �  �  �  8  �  J  h  �  N  �  Z  �  �  t  �  �  �  �  3  �  �  �  i  m  �  �  @<  @<  @<  @<  @<  @<  @<  @<  @<  @<  @<  @<  @<  @<  @<  @<  @<  @<  @<  @<  @<  @<  @<  @<  @<  @<  @<  @<  @<  @<  @<  @<  @<  @<  @<  @<  @<  @<  @<  @<  @<  @<  @<  @<  @<  @<  @<  @<  @<  @<  @<  @<  @<  @<  @<  @<  @<  @<  @<  @<  �  �  �    J  a  b  Y  F  -    �  �  �  B  �  s    �    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  |  l    p  P  �  8  3  �  _  �  �  �  �  4  �  �  s  �    
	  �        �  �  �  �  �  �  �  �  �  �  u  d  L  2  
  �  �  6  '      �  �  �  �  �  �  f  B    �  �  �  s  $  �    	    �  �  �  �  �  �  �  �  �  �  �  ~  j  O  1    �  �  �  B  �  �  �  G  �  �  �  �  �  �  �  x  B  �  l  �  =  :  d  a  Q  G  Z  O  @  .    �  �  �  {  L    �  �  �  �  S  +  #          �  �  �  �  �  �  �  �  �  r  a  S  D  5  O  N  L  J  H  C  4  %  
  �  �  �    P  !  �  �  A  �  �  ;  F  O  R  P  K  @  5  $    �  �  �  �  �  �  u  i  s  �  b  �  R  O  K  F  B  :  0  !      �  �  �  �  �  ~  C            �  �  �  �  �  |  c  J  1       �   �   �   �   �   x  �  (  C  P  R  D  &    �  �  j  )  �  �  +  �  �  	    �  �    K  �  �  T  �  �    B  S  8  �  �    ~  �  �  �   �  n    O  �  �  �    5  <  1    �  �  q    �  �      p  �  �  �  �  �  �  �  �  �  �  �  �  r  I    �  `    �  �      �  �  �  �  �  H  
  �  �  I  �  �  X  �  �    i  �  �  �  �  �  �  �  �  �  �  �  x  i  [  V  c  p  }  �  �  �    �  �  �  �  �  �  �  �  p  Q  8    �  �  m    �    �  �  �  �  �  �  �  �  �  �  �  �  �  a  2    �  m    �  �  �  �  �  m  O  2         �  �  �  �  c  !  �  o    �  �    �  �  �  �  �  �  �  �  �  g  L  +    �  �  �  E     �          �  �  �  �  i  l  L    �  �  a    �  Q  �  �  F  D  B  A  ?  >  <  :  9  7  >  M  \  l  {  �  �  �  �  �  7  ?  A  =  1  !    �  �  �  �    S  #  �  �  �  U    �  w  �  �  �  �  �  �  �  �  �  �    ]  9    �  �  �  y  Q  T  �  �    7  E  A    �  �  h  !  
�  
m  	�  	  �  �    "  �  �  �  �  �  �  �  �  �  _  6    �  �  �  `  $  �  �  e    $      !    	  �  �  �  �  W    �  �  2  �  �  �   �  �  �  �  �  �  �  �  �  �  �  �  l  (  �  �  >  �  �  <  ,  &  ,  -  +  $      �  �  �  �  V  &  �  �  �  H    �  S  U  A  .      �  �  �  �  �  �  �  �  ~  Y  2      P  �  �  
                 "  :  L  \  e  \  �  �  �  �  �  o  �  �  �  �  �  �  �  �  �  �  
  �  �  {  3  �  U  �    n  b  U  K  @  -  "  '  ,  /  1  0  2  6  B  R  u  �  �  �  �  �  �  h  9    �  �  r  8     �  y  $  �  ^  �  :  �  �  �  �  �  �  �  �  �  �  �  }  W  0    �  �  �  l  (  �  b  �  �  �  �  j  Q  4    �  �  �  |  K    �  �  Y    �  �  �  �  �  �  �  �  �  �  g  C    �  �  �  w  J    �  �  �  a  y  u  m  e  \  [  \  [  M  ?  -    �  �  �  �  o  
  o  �  �  �  �  �  �  �  �  �  �  �  �  �  �  m  @  �  �  K  	  �  �  �  �  �  t  \  B  ,    �  �  �  �  L  	  �  v  -   �  N  F  =  6  0  +  !      �  �  �  b  =     �   �   �   d   5    
    	  �  �  �  �  j  A  %  �  �  �  J    �  �  �  c  �  �  m  8    �  �  �  Z  '  �  �  =  �  8  �     f  �  2  '      �  �  �  �       �  �  �  �  �  �  y  b  K  3    �  �  �  |  i  U  @  ,    �  �  �  �  �  �  h  A    �  �  �  �  �  j  ?    �  �  �  k  N  <  )    �  �  �  �  w  ;  �  �  �  �  �  l  R  5    �  �  q  3  �  �  x  Y  >  %    i  N  F  G  A  8  '    �  �  �  �  |  J    �  �  N  �  ,  �  �  �  r  ]  E  )  
  �  �  �  F  �  �  X    �  E  �  @  =  L  R  I  /  	  �  �  %  �    R  �  �  
�  	�  I  �  J  �  �  �  j  I    �  �  �  E    �  �  �  �  �  A  �  �  q  (    �  �  �  �  _  0    �  �  �  Y  ,  �  �  �  �  q  N  )  �  �  U    �  �  �  H    �  �  l  #  �  �  9  �  �  �  F  U  7      �  �  �    .  C  A  -    �  �  �  H     �  j  8  .  $    
     �  �  �  �  �  Y  %  �  �  J  �  n  �  �  $  5  ,    �  �  �  �  {  W  2    �  �  �  Z    �  �  /  1    �  �  �  �  \  ;    �  �  �  @  �  �  L  �  �    c