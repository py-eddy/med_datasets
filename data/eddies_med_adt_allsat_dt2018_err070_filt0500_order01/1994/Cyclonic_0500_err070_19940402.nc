CDF       
      obs    Q   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?�t�j~��     D  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M��\   max       P��     D  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ���   max       <�C�     D   4   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?!G�z�   max       @F\(��     �  !x   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ���Q�     max       @vm�����     �  .    effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @/         max       @Q@           �  :�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @ʩ        max       @�_�         D  ;l   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ��S�   max       <u     D  <�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�[�   max       B4�	     D  =�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A��M   max       B4�C     D  ?8   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >̢   max       C�5     D  @|   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >��   max       C�)     D  A�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          L     D  C   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          ;     D  DH   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          9     D  E�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M��\   max       P�ԡ     D  F�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?���{���   max       ?�֡a��f     D  H   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ���   max       <�C�     D  IX   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?!G�z�   max       @F�z�G�     �  J�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ���Q�     max       @vk
=p��     �  WD   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @'         max       @R`           �  c�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @ʩ        max       @�          D  d�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         ?"   max         ?"     D  e�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�*�0�   max       ?����E�     P  g            
   	               	                        1                                 3   7            -                                    	      !         4   	                  8   L                                                          
         OL�NlE\N/�gN��NʿQN��N��
N�N�P�1N��4N/qrO
�N�y^M��\N=�MNg�P�P9��NQ�nN2��O�^WN�t:O�
BN���OuF�N��N�P�N�/�P6RPO���O~/�O2�{PX�;P��O�HtN��5N ��Oz>mN���Ob�Om�OE�O�n�N�fOK��N3�OV�@O��N�l�NFPHO���N�56OWj�O���N=� N�SPOPC��O���OR�O�@O{�N�(�N�mRN��N�[�Ow�UO���N���OH��O�ƞN.��OVY�M��\N5<�M��,N��HOM�N���N��N�<�C�<�C�<�o<e`B<#�
;��
;��
;D��;D��$�  ��o��o�o�ě���`B�t��49X�49X�e`B�e`B�e`B�u�u�u�u�u��o��C���C���C���C���t����㼛�㼬1��1��9X��9X��j�ě����ͼ��ͼ��ͼ��ͼ���������/��`B��`B��`B��h��h���o�+�+�+�\)�t��t��t����#�
�,1�0 Ž0 Ž0 Ž0 Ž49X�<j�@��D���L�ͽL�ͽL�ͽaG��m�h�y�#�y�#��%�������������������������������������������,/;?HNH;/),,,,,,,,,,!'( 1<ITUVbegdbUI<<31111����������������������	
 
����������#"!"
������������������������������������������� #./2<?@</%#        knpz���������}zxyrnk-/2;FHHNQTUTOH;/+*--	

								���"��������}�����������~��~tr}JTg�����������gNB;9J�$����������.5>BNSQNGB<5........ *6CMQTYZOC*� ����

��������� #/39=;53+
�����


�������
#HUXWTPH</)#
TUacnpppnla`YUTTTTTTY[fgt����tsg\[YYYYYYcgt��������tog`]ccccz����������������zxzx��������������}vrsxBOQFO[t��������h^B?B��������������������]bcg������������zWU]
#H��������{U<0

SXcz����������zaZSQS��"#)/2/.#
����������������������)*69@E[hqxvqrqh[XO6)STZaimtxvxmla[TSSSSS������������������������������������������������������������������������>BHOPY[_gf[ONBB?>>>>ty���xtqh[URRTV\dht)66;6)"��������������������HUaju|��������maUHEH���������������������������������������������!"�������������������������������������������pt����������������vp����������������������������������������)25BN[a_[TNB51)&�����"�������������(4:5(�������?BO[ht��������th[OB?��������������������./;<HU__`]ULH<://-..����	

����������#&-0230.#!���������������������
#$%#!	���������������������������)6BVfhc[B6)��S[dgtv}|xtgc[USSSSSSdgt��������������gcd�
#0;<C<9;0#
������������������������HJRUanz������znaSIH��������������������#0590%#��������������������dgjtt������thgddddddfgkt�����������}tmff��������������������"#&-/9<HD<84/&#" ! ":<HLRHD<;8::::::::::�z�n�a�_�U�M�J�U�a�zÇÓÜàâäàÓÇ�z�#� �����#�+�/�2�<�E�<�/�#�#�#�#�#�#����������������������������������������ֺκɺ��ƺɺֺ������޺ֺֺֺֺֺּ����������������ʼּټ߼߼ּּʼ��������U�U�U�U�Y�Z�a�h�n�t�zÁ�|�z�n�a�U�U�U�U���������������������Ǽɼ����������������A�=�4�(�$������(�4�>�A�J�M�U�M�A�A�׾����s�f�P�Y���������� �������׾����������������������ʾϾѾ;ʾ��������H�A�<�7�<�A�H�U�W�[�U�Q�H�H�H�H�H�H�H�H��������þü����������������
��������������������������������������������������$�#�$�(�/�0�=�D�=�:�1�0�$�$�$�$�$�$�$�$�Z�Y�U�Z�f�s�����|�s�f�Z�Z�Z�Z�Z�Z�Z�Z������������������������������������������������0�H�T�a�x�����z�m�a�H�;�"������������������ſӿѿ�����������ݿ����������������������������������������������������������������������������������;�5�"�	�������	�"�;�G�M�M�J�J�M�R�G�;�����������������������������������������N�B�)���	�
��)�B�[�g�t�t�N����������������������������������������� �'�&�)�6�B�O�[�j�i�[�S�O�G�.�)��m�m�j�m�x�z�������������z�n�m�m�m�m�m�m�������������������	�	��	����������������������	�������	������𺼺������ֺ���!�-�F�a��x�l�S�:��ɺ������������������ûл��� �����ܻлû��Q�P�T�`�f�e�[�`�h�y���������������x�m�Q�6�4�+�(�*�6�?�B�E�O�[�^�e�h�k�f�[�O�B�6�a�T�;�/�*���=�a�z�������������������a�����o�A���5�Z�s�����������������������(�������(�5�N�g�s�������h�Z�A�5�(²±²·¿��������������������������¿²�����������������Ŀ�������������������������߹ܹϹù������ùϹܹ�������������ƻƳƬƧƤƧƳ�������������������������.�"��	������	��.�:�;�L�[�c�`�T�G�;�.ĿĹļ������������#�0�<�0�%�
��������ĿŔōňŋŐŔŚŠŧŭŹ����������ŹŭŠŔàÓËÇ�z�t�o�vÀÇÓÞàãôüÿùìà�l�b�_�]�_�l�o�x���������������x�l�l�l�l�z�����������Ľͽн��ݽнĽ��������y�zìââììù����ùöìììììììììì����������������.�D�G�C�6�+�%������������ùϹ�����'�?�4�'�����ڹù������ܹԹϹùϹعܹ������������������������������������������������������׾ʾ������ʾ׾���������	�����׿y�r�m�b�`�]�`�m�y�������������������y�y�s�h�Z�W�T�]�g�o�s���������������������sŹŮŖŇ�{�}ŇŠŭŹ������������������Ź��������������������������������������������������$�$�&�$�������������������������������������������������Y�M�a�|���ּ����%�&����ּ�������Y�4�������4�@�M�Y�h�m�m�k�f�Y�M�@�4�M�L�F�G�F�F�F�J�Y�f�r�|�z�v�t�t�q�f�Y�M���x�r�d�Z�W�Y�^�l�x��~���������������������߾ؾؾ����	������
�	�������������
��!�)�-�1�-�!���������������������(�4�4�7�4�(� �����D�D�D�D�D�D�EEEEED�D�D�D�D�D�D�D�D�EEEE!E*E.E7EAECEPEZE\EcE\EPECE>E7E*E�s�g�_�U�Y�b�s�������������������������s������z�z�|������������������������������������)�6�<�7�6�2�)�������/�+����������#�/�H�Q�Y�U�B�<�/�������������л׻ܻ���������ܻл��ʾ¾����������ʾ׾��׾ʾʾʾʾʾʾʾʿĿ������������������Ŀѿտݿ���ݿѿĽннͽнݽ�����ݽнннннннннн���������������������������������������������������������������������������¿²²ª²¿������������������������ĿĹĳĬĦģĦĳĹĿ������������������Ŀ����������������������������������������E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D� E j D $ J Q 4 � W * 1 3 / | D O K . E _ P ` M W Z n H 7 a 1 � % , M V } V J 4 h { ) ` H U ^ . ~ f Y 4 5 5 1 I * Q g * n i  J L 8 W % ] . 8 8 r D Q k d 4 + � h @    �  �  ]  �    �  �         O  >  �  Y  f  �  i  :  z  s  �  R  �  	  C  �  �  �    �  �  z  �  �  �    [    �    f  �  L  �  �  r  �  J  �  �  =    �  �  e  �  [  8  �    �  @  �  �  �  0  �  �  �  �  �  [  �    \  N  �  �  �  A  &�T��<u<T��;�o:�o�#�
$�  ��o��j�t��o�D����`B�t��T���e`B�#�
�u���㼋C�����1��㼬1��㼋C����
��/��7L��hs��h��P�'�o�49X�+�ě��D����`B�,1�t��,1�D���t��0 ŽC���P�y�#���+���
��P�,1�L�ͽt��#�
��㽺^5��S��y�#�y�#�T���49X�]/�q�����P��\)��\)�aG���+��C��P�`���P�Y��]/�y�#��o��hs��7L����
B:\B6RA�[�BpB&�Bh�B$QB�,B �LB4�	BE�B�MA���B��B	BtMB��B	ǊB�PB�B/��B;�BeyB�BX�B8gB	z�B	ޮB��BxRBиB�=B
�B&��A�1B2.B��B�~A��B�BWtB��B��B٬BʆB%PB6�B7�BȫB��BlB*�)B'�B�B �iB�wB�B-'IB��B(B �B*�B#�uB%J�B��B�BS�B��B	J�B
e�B$�B8�B��B"�(B%H�B<zB	�AB
U5B�BP�BBAB��A��MB�B&��B�bB$O%B@B!��B4�CBA~B�A���B<�B@B�TB��B	�BDhB�B02%B�B��B�nBW>BIB	toB	�lBDBA�B��B�UB
׈B&�A�\B:6BF�B?�A���BA�BF�B�[B��B��B<�B�BBHB�B��B��B+MB*�B(�B?�B Y�BвB��B-<�BH@B�B ��B?�B#�B%?�B��B�(B@hB9�B	@SB
�{B$I_B?KB��B"YB%a	B9�B
>uB
@LB��B@(B��AȘA�?A�X$@>��@��FA�&@�~�A7vAL�AM��AľdA�<�A�/B
<�AAͭA�~>A���AyM�AH��AHR�A_?�A�,A��aA��A�/�A���A�axAY�@]i�@���Amv�A�[A�lkA�0�A��A��Av��>��%B�.A`��A��A�lUA�<�@��kA#e#A�� A�E�>̢?:0A�d6AUG�An[�A��5A�MKB��B�UA���@���@шs@���@�UAX�/@a��A4��C�IC���A�ylA��lA�
�A��N@��AP�!AyCA*�fA#S�B��A� �A㕚A�7bC�5C���AȐ�A��DA�i�@<_@��yAǀ�@�`�A3'�AK3�AMr�A�[�A�kA�G�B
�;AA	�A�YA��VAy�xAH��AI�A\��A��A�}�A���A�]�A���A���AY�N@Td@��Al��A؀YA���A��A��A��Av��>��B�KA]#BA�A��\A�A.@���A!C�A�y�A���>���?�8A���AT�{Am��A�$A��[B��B�A��A*�@�0�@��\@�pAY�@a�'A5jOC�H C���A��SA��!A�oAÅl@�~�AR�Ay~A+��A"��B�A��`A�|A�+C�)C��               
               
                        2                                 3   7            .                                    	      "         4   	                  9   L                                                                                              -                        )   -               %                  3      #      1   ;   !                                       '         #         !            7   !                                    #                                                         )                                                            /            #   9                                                                        7                                                                     O	z�NlE\N/�gN�|N�,4N;��N��
N_QtO�gN�d�N/qrN�hN�;SM��\N=�MNg�O>v�O#a�NQ�nN2��O^�^N�0�O�CcN���O��N��N�P�N�/�P.sO�HN܎�O%��O٘SP�ԡOC�N��5N ��Ob�N���Ob�Om�N�HcO�n�N�fO)�/N3�OV�@O��N�l�NFPHOAncN�56N���O}�&N=� N���N�ZpPC��O}\�OR�O8��Nޓ�N�(�N�mRN��N�׶O0�\O���N���N��oOdCHN.��OVY�M��\N5<�M��,N��HO&��N���NubJN�  �  �  r  x  z  l  �  �  @    �  �  �  <  �  �  Y    z  �  �  I  i  (  M  F  �  �  K  t  j  �  �  �    �  �  �  \  �  �  !  �  u  �     �  �  �  �  �  X  �  W  �  �  J  �  
G  -  �  b  C  �    	�  �  �  �  �  �  �    i      �  �  s  �  �<#�
<�C�<�o<T��<t�;o;��
;o:�o�o��o�D���D���ě���`B�t��ě��'e`B�e`B��C���o��C��u��1�u��o��C���9X��j��1�����/���
�o��1��9X�ě���j�ě����ͼ����ͼ��ͼ�`B������/����`B��`B�P�`��h�C��t��+�C��C��\)�e`B�t��,1�#�
�#�
�,1�0 ŽD���D���0 Ž49X�P�`�T���D���L�ͽL�ͽL�ͽaG��m�h�}�y�#��C��������������������������������������������,/;?HNH;/),,,,,,,,,,	 &'4<IUbcfbbUI@<5444444����������������������	
 
���������"! ���������������������������������������� #./2<?@</%#        sz���������zuossssss./5;HKMQJH;//-......	

								���"����������������������������Z[egt���������tg\[YZ�$����������.5>BNSQNGB<5........"*6CIMNOPPOC6*����

����������
#/69;942)#
�����


�������#/<@HPPKHC</-%#TUacnpppnla`YUTTTTTTY[fgt����tsg\[YYYYYYcgt��������tog`]ccccz~����������������|zv{���������������xuvahtt�����������tnhaa��������������������cgt�������������td`c
#Ir�������{U<0	
_aemz�������zrmga`^_��"#)/2/.#
����������������������-6;AGO[hovtopoh[O6,-STZaimtxvxmla[TSSSSS������������������������������������������������������������������������>BHOPY[_gf[ONBB?>>>>X^fhtv����}tmh[UTUVX)66;6)"��������������������HISUYalntxyuna[URMHH�����������������������������������������������������������������������������������������������������������������|yu{�����������������������������������������)5BN[[[RNJB53*)�����"��������������!%& �����?BO[ht��������th[OB?��������������������//1<DHUV\]ZUH<1/////����	

����������#&-0230.#!��������������������
""
���������������������������)6BVfhc[B6)��S[dgtv}|xtgc[USSSSSSrt����������ytkkrrrr���
#-021*#
������������������������HJRUanz������znaSIH��������������������#0590%#��������������������dgjtt������thgddddddgnt����������vtoggg��������������������!#$*/:<?<850/-##!!!!:<HLRHD<;8::::::::::�n�e�a�Y�V�a�n�zÇÓÔÝÝÓÇ�z�n�n�n�n�#� �����#�+�/�2�<�E�<�/�#�#�#�#�#�#����������������������������������������ֺӺɺ��Ǻɺֺ������ٺֺֺֺֺֺּ������������ʼּ׼ݼּܼԼʼ������������n�g�a�\�a�a�a�]�a�e�n�s�z�{�z�u�n�n�n�n���������������������Ǽɼ����������������A�@�4�(�%���	���(�4�:�A�F�M�N�M�A�A�������s�f�T�\������پ����������׾������������������ɾʾξʾɾ��������������H�A�<�7�<�A�H�U�W�[�U�Q�H�H�H�H�H�H�H�H��������ÿ�����������������������������������������������������������������������$�#�$�(�/�0�=�D�=�:�1�0�$�$�$�$�$�$�$�$�Z�Y�U�Z�f�s�����|�s�f�Z�Z�Z�Z�Z�Z�Z�Z�����������������������������������������;�/�"�%�/�;�F�H�T�V�a�h�m�q�m�h�a�V�H�;�������������������Ŀѿٿݿ��߿ѿοĿ����������������������������������������������������������������������������������.��	���������	��"�.�;�<�E�E�D�D�F�;�.�����������������������������������������[�N�B�2�����)�5�B�[�g�t�t�[��������������������������������������)�'�$�&�(�)�-�6�B�O�S�[�d�_�[�O�M�B�6�)�m�m�j�m�x�z�������������z�n�m�m�m�m�m�m�������������������	�	��	����������������������	�������	�������ɺ�������������!�-�F�S�f�k�S�-����ɻû����������������ûлܻ���������ܻлÿy�s�m�m�g�f�m�p�y�����������������~�y�y�B�6�6�,�)�+�6�B�O�[�]�c�h�j�h�d�[�O�B�B�m�c�T�I�L�F�E�S�a�z�����������������z�m�����p�B�(��5�Z�s�����������������������5�/�(���5�6�A�N�Z�g�i�l�g�c�Z�O�N�A�5²±²·¿��������������������������¿²�����������������Ŀ������������������������ܹϹù��������ùϹܹ��������������ƻƳƬƧƤƧƳ�������������������������.�"��	������	��.�:�;�L�[�c�`�T�G�;�.ĿĹļ������������#�0�<�0�%�
��������ĿŠŘŔŒŔŖşŠŭŰŹž��������ŹŭŠŠàÓËÇ�z�t�o�vÀÇÓÞàãôüÿùìà�l�b�_�]�_�l�o�x���������������x�l�l�l�l����������������������ɽн۽ݽнĽ�����ìââììù����ùöìììììììììì����������������.�D�G�C�6�+�%����͹ù��������ùŹϹܹ�����������ܹϹ͹����ܹԹϹùϹعܹ������������������������������������������������������׾Ҿʾž��¾ʾ׾����� �	�������׿y�r�m�b�`�]�`�m�y�������������������y�y���u�s�g�_�Z�d�g�s����������������������ŞŐńŊŔŠŭŹ������������������ŹŭŞ������������������������������������������������"�$�%�$�������������������������������������������Y�M�a�|���ּ����%�&����ּ�������Y�@�4�%�����!�'�4�@�M�Y�^�f�g�d�Y�M�@�M�L�F�G�F�F�F�J�Y�f�r�|�z�v�t�t�q�f�Y�M�x�v�j�c�`�`�e�l�v�x�������������������x��������������	����	�������������������
��!�)�-�1�-�!���������������������(�4�4�7�4�(� �����D�D�D�D�D�D�EEEEED�D�D�D�D�D�D�D�D�E*E#E$E*E1E7ECEPE\E]E\EZEPECE:E7E*E*E*E*�����s�g�c�Z�^�g�n�s��������������������������z�z�|������������������������������������)�6�<�7�6�2�)�������#� ���#�/�/�<�C�H�T�M�H�C�<�/�#�#�#�#�лû����������û˻лܻ���������ܻоʾ¾����������ʾ׾��׾ʾʾʾʾʾʾʾʿĿ������������������Ŀѿտݿ���ݿѿĽннͽнݽ�����ݽнннннннннн���������������������������������������������������������������������������¿²²ª²¿������������������������ľĳĮĩĦĳĿ������������������������ľ����������������������������������������E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D� @ j D " K e 4 � W # 1 ! 3 | D O "  E _ A U H W @ n H 7 Y , X & ( F G } V H 4 h { ' ` H N ^ . ; f Y  5  & I  T g $ n `  J L 8 M  ] . ' # r D Q k d 4 3 � \ @    *  �  ]  �  �    �  �  �  �  O  �  �  Y  f  �  �  \  z  s  �    m  	  Y  �  �  �  W  B  $  a  �  �  a    [  �  �    f  �  L  �  �  r  �  ;  �  �  �      �  e  �  *  8  �    �  �  �  �  �  �  r  �  �    �  [  �    \  N  �  l  �  �  &  ?"  ?"  ?"  ?"  ?"  ?"  ?"  ?"  ?"  ?"  ?"  ?"  ?"  ?"  ?"  ?"  ?"  ?"  ?"  ?"  ?"  ?"  ?"  ?"  ?"  ?"  ?"  ?"  ?"  ?"  ?"  ?"  ?"  ?"  ?"  ?"  ?"  ?"  ?"  ?"  ?"  ?"  ?"  ?"  ?"  ?"  ?"  ?"  ?"  ?"  ?"  ?"  ?"  ?"  ?"  ?"  ?"  ?"  ?"  ?"  ?"  ?"  ?"  ?"  ?"  ?"  ?"  ?"  ?"  ?"  ?"  ?"  ?"  ?"  ?"  ?"  ?"  ?"  ?"  ?"  ?"  �  �  �  �  �  �  �  �  �  �  �  w  0  �  u    {  �  �  �  �  �  �  o  ]  K  9  (      �  �  �  �  �  �  �  �  �  �  r  j  b  Z  R  J  B  =  ;  9  7  5  3  2  6  :  >  B  E  I  w  x  w  r  k  ^  O  9      �  �  �  u  L  "  �  �  Y    w  y  z  y  y  v  s  n  i  a  V  I  :  )             �  '  @  U  d  l  m  l  w  �    �  �  �  �  �  �  �  Q  �  ,  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �      1  G  R  \  d  g  i  l  o  s  v  y  |  7  ?  <  4  2  4  7  <  ?  >  2    �  �  �  >  	  �  �  i  �  �  �            �  �  �  �  �  �  �  �  j  L  /      �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �    `  8    �  �  �  �  �  �  �  �  �  �  �  �  x  Z  =  #    �  �    �  �  �  �  �  �  �  �  �  �  �  �  �  n  U  <     �   �   �  <  2  '        �  �                  �  �  �  �  �  �  �  �  �  �  �  |  ^  @    �  �  �  {  O  %     �   �  �  �  �  �  �  �  �  z  i  Q  :  #  	  �  �  �  �  y  [  =  z  �    "  4  ?  @  6  ;  X  N  =  !  �  �  �  <  �  �  �  .  |  �  �  �  �  �         	        �  �    �  �  >  z  v  q  m  f  U  D  3       �  �  �  �  �  ~  R  !   �   �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  v  c  P  =  �  �  �  �  �  �  �  �  �  �  �  �  n  Q  ,    �  �  p  N  =  A  E  I  B  :  2  (        �  �  �  �  �  �  �  �  s  N  e  f  Y  F  /    �  �  �  �  [    �  �  P  )      �  (  #            �  �  �  �  �  �  �  �  �  �  �  �  z  �       7  K  M  H  @  1    �  �  �  8  �  �  M    �  �  F  E  E  D  C  B  A  @  ?  ?  9  /  $         �   �   �   �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  ~  l  Y  F  2      �  �  �  �  ~  ^  =    �  =  C  J  5    �  �  �  �  ]  ,  �  �  &  �  |  "  �  6  Z  I  i  t  l  [  F  1    �  �  �  T    �  7  �  �    E  8  �  �    ;  O  ]  g  i  f  ^  S  C  -    �  �  �  �  �  _  �  �  �  �  �  |  s  h  U  G  ;  2  &    �  �  �  q  ,  J  �  �  �  �  �  �  �  �  �  �  �  �  ~  a    �  �  �  }  !  w  }  d  ?    �  �  {  L    �  �  W    �  n     �  I   �  �  �  �  �  �  �  �  �      
  �  �  �  w  8  �  �  v  '  �  �  k  A    �  �  �  �  �  �  n  9    �  �  t  �  r  �  �  �  �  �  �  �  �  �  �  �  �  �  �  }  q  e  Y  M  A  5  �  �  �  �  �  �  �  �  z  Q    �    \  O  9  &    �  �  \  V  P  J  D  9  /  $      �  �  �  �  �  �  q  T  7    �  �  �  v  [  P  4    �  �  �  �  �  y  2  �  �  0    i  �  �  �  s  Z  >  $  	  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �        !      �  �  �  �  {  E    �  �  F  �  �  �  �  �  �  �  ^  0  �  �  �  8  �  �  e  a    �  �  0  u  q  i  Y  C  )    �  �  �  �  ^  '    �  �  H  �  �  /  �  �  �  �  �  �  �  �  �  �  �  p  F    �  �  t  >  	  �           �  �  �  �  �  �  �  �  �  �  �  �  z  f  O  9  �  �  �  �  �  �  �    i  Q  <  '    �  �  �  �  �  g  ;  �  �  ~  H  �  �  �  �  �  s  L    �  �  W    �  <  �  =  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  y  m  a  U  I  �  �  �  �  �  �  �  �  t  d  T  D  1      �  �  �  �  �  �  �  �    @  g  �  �  �  �  �  �    P  �  {  �    2  N  X  S  M  G  A  :  2  )        �  �  �  �  �  ~  [  6    �  �  �  �  �  �  �  �  �  �  �  �  �  �  z  K    �  z  	  A  K  Q  T  V  U  N  C  5  $    �  �  �  �  _  1  �  �    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  
    1  �  �  �  �  �  �  �  �  �  �  s  c  R  A  /      �  �  �  B  D  F  H  I  A  :  3  (      �  �  �  �  �  �  �  �  �  �  �  �  �  q  T  '  �  �  �  X    �  �  E  �  i  �     �  	w  	�  	�  
#  
=  
F  
@  
%  	�  	�  	t  	%  �  [  �  �  !  G  x  �  -  (    
  +  &      �  �        �  �  {  /  �  �  g  �  �  �  �  �  �  �  �  p  V  8    �  �  �  x  )  �  �  E  +  E  X  a  `  Z  Q  H  ?  3  $    �  �  �  z  C    �  �  C  >  9  4  /  *  $                  �  �  �  �  �  �  �  t  a  G  *  	  �  �  �  g  8    �  �  h  0  �  �  "    �  �  �  �  �  i  G  &    �  �  �  R    �  B  �  �  �  �  	�  	�  	�  	�  	�  	�  	h  	?  	  �  t  )  �  �  T        5  �  �  �  �  �  �  �  �  V  &  �  �  �  O  �    !  �  J   �  �  �  �  �  �  �  �  s  Q  *    �  �  �  �  I  �  �  �   �  �  �  �  �  �  �  �  �  }  `  B  #    �  �  �  �  �  y  O    `  �  �  �  �  �  �  �  k  O  /    �  �  �  d  5  (  -  L  L  J  v  �  �  �  �  m  X  B  -    �  �  �  t  )  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  q  D     �   �   �   g        �  �  �  �  �  �  �  �  �  }  O    �  �  _    �  i  ^  S  H  >  3  (         	    �  �  �  �  �  �  u  ]        �  �  �  �  �  �  �  �  �  �  �  �  �  �  z  l  ^        (  +  -  /  +  $      	  �  �  �  �  �  �  �  �  �    y  s  m  f  `  Y  Q  J  D  A  >  ;  ;  ;  <  C  J  Q  �  �  �  �  �  �  x  Q  0    	    �  �  �  �  �  �  �  �  s  o  j  f  h  l  p  n  i  d  Z  L  >  /       �  �  �  Q  g  �  �  �  �  �  �  �  �  _    �  _    �  W  �  �  R  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �