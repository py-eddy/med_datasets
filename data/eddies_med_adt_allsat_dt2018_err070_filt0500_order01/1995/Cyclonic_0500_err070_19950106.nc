CDF       
      obs    G   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?��t�j~�       �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M۲   max       P��       �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ��^5   max       =��-       �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?�\(�   max       @F��G�{       !    effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�fffff�   max       @v�\(�       ,   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @-         max       @Q@           �  70   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @ɖ        max       @�o@           7�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �	7L   max       =�t�       8�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A��_   max       B5'g       9�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A��I   max       B5>t       ;   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       ?��I   max       C���       <0   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       ?�%   max       C���       =L   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          `       >h   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          5       ?�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          -       @�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M۲   max       P6"       A�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?���
=p�   max       ?�K]�c�A       B�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ��^5   max       =��-       C�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?�\(�   max       @F��G�{       E   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?��G�{    max       @v�\(�       P(   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @'         max       @Q            �  [@   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @ɖ        max       @���           [�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         @9   max         @9       \�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?������   max       ?�K]�c�A     0  ^                                    `         !         *                        ,         	   *         '            %      '            
             $                                 C            	      A             &         ,N9�7N�1M۲N_<�N��N�<kN5O:�O�[O<��NA�P��O�R�NcxO<�N��Od��O��O?M�Z�NI�INF��O��NE6)N��Ou��N�P�BN���P)Z4O'��O&��PC��NU�N��QO��>Oh�\N?�)O��NM��N�1{Or�Oa�!O �N KsO��O�")O�{OX�O��O��N�A.NK�IN�FN�c�O)I�OZa;P^O��OW��N�?�N���OhZO�	TO��NIX�O�}WO��N�[4N�_�O�=��-<D��;�o%   %   �o��o���
�ě��#�
�49X�D���D���T���T����o��o��o��t���t���t����㼛�㼣�
��1��9X��9X��/��`B�o�C��C��\)�\)�\)�t��t���P��P����w��w��w�'0 Ž49X�8Q�8Q�@��D���H�9�L�ͽL�ͽL�ͽL�ͽT���T���]/�aG��aG��ixսixսu��7L��O߽�\)���P���T���罰 Ž�^5��������������������).5ABDCB5)���������������mmz|�����zwrommmmmmm�
#/5<<<;7/#
��y��������������{~}y��������������������)3200.,)
@BNV[agkt{tg_[NB?99@9<HU[`cdbca\UH<84349��������������������kz�����������|vqfcck#/<HUWazzn[UE<//##��������������������������������������������	
������������#,-(#
� �������� 
#3HONI<# ������������������������������������������)-66:6)'��������������������	)/58BGNSB5)'	�������������������������������@BO[w�����yth[OGB?=@��������������������6Bhpss��zth[OB71206{������������{{{{{{{Wabt������������g[UW��!)(������ktx��������������tpk�5N[lg_aZNB1����RUYabhlkaZUSRRRRRRRR#'&$��5:7BNYgt������t[N=55�������������������������������������������������������������������������������������������������� #/3<HTUYZUSH</#  ��
!*&+060,#
����
#(.-//4/#
  ����������������������������������������#/EHUZXRSHD1#
]az��������znka\b^W]���������������!')06BO[VQOKBA6)����

��������������������������������������������������������������������������������������������������������������������������������#*/<N]gjomaUH<6.(" #')5BBFECB>5*))!''O[hnpsppnnoiga[URKJOtz���������}znttttttrt}�����������wtqorr����	������������ )0*
������BN\��������tgVNB@B>B��������������������GPUbn{������{nb^UKGG��������������������Wanyz�������|zna[ZWW8<IJRUVUUSMIGDA=?<88<BEGEA<1//'##"! "#/<E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E��������������$�+�+�'�$�#���������������������������������������������������<�:�8�<�I�P�U�b�h�d�b�U�I�A�<�<�<�<�<�<ŹŲŭŬŧŧŭŸŹ������������������ŹŹ�Y�M�M�L�M�Y�f�k�r���������������r�f�Y�#������!�#�/�3�/�)�#�#�#�#�#�#�#�#�ѿ˿ſƿѿݿ������#�&�������ݿ������������������������������������������N�K�D�F�N�Z�g�s�����������������s�g�Z�N�H�H�C�A�H�S�U�Y�X�U�H�H�H�H�H�H�H�H�H�HƎ�y�t�{�uƚ�����$�1�8�9�0��������ƧƎ�4�*�&�+�5�<�H�S�V�c�zÓÓÎÐÇ�w�U�H�4�b�\�a�b�n�{��{�u�n�b�b�b�b�b�b�b�b�b�b����������(�A�I�P�K�B�5�������`�\�T�P�T�T�`�m�y�����������y�m�`�`�`�`ìáÙÓÖàìù��������������������ùì��������	����)�6�B�O�X�Q�O�B�6�������������������������������������������s�s�j�g�Z�X�Z�g�k�s��������s�s�s�s�s�s�M�K�@�=�>�@�K�M�Y�b�[�Y�S�Q�M�M�M�M�M�M�A�5�5�4�5�A�A�N�U�W�N�I�A�A�A�A�A�A�A�A��
��	���)�2�5�>�B�F�H�D�B�5�3�)�������'�4�:�@�4�'�����������a�_�]�a�d�l�n�q�z�|�|ÇÓÕÓÇ�z�n�a�a�l�i�`�d�x���������������������������x�l�����������������������������������������#����G�T�p�y�����������������m�T�;�#�Ŀ����������Ŀѿݿ��ݿܿѿĿĿĿĿĿ��t�U�U�d�l�t¦²��������¼º²���������������ʾվ׾ݾ���׾Ҿʾ������b�Z�V�I�C�A�I�M�V�b�o�x�{�}�|�{�s�p�o�b�������������������*�C�O�S�O�G�6�*�����	�	��	��"�.�:�.�+�"���������f�e�^�Z�R�Z�f�s�����������������s�f�f���׾ϾԾ��������	�������	�������ܾ׾Ҿ˾ʾ׾����������	���׾;ʾ������������ľʾ׾߾���׾׾׾׺@�<�=�>�@�F�L�[�r���������������~�Y�L�@�f�c�f�j�s�������s�f�f�f�f�f�f�f�f�f�fD�D�D�D�D�D�D�D�EEEED�D�D�D�D�D�D�D߿����������������������������������������!����!�:�F�S�\�_�e�l�s�u�k�_�S�F�-�!E7E0E4E7E>ECEPE\EiEjEuE�EvEiEeE\EZEPECE7����������������������������������������ā�t�g�b�e�h�tāčĚĦĿ������ĳĦĚčā�Z�M�F�G�M�Q�O�Z�f�s����������������f�Z��}�}�����������ʾ׾߾߾۾׾¾���������ּʼüɼӼ������#�'�����������m�k�a�T�P�H�C�;�3�;�H�T�Y�a�l�m�u�v�o�m���������$�0�=�G�I�N�I�?�=�$���ŭŪŠŔŉŔŘŠŭŹ��������žŹŭŭŭŭ�I�H�I�M�U�b�m�n�{ń�{�n�b�U�I�I�I�I�I�I�O�O�C�A�C�J�O�\�h�h�u�v�u�h�h�\�O�O�O�O���������������������������������������������	�
���#�'�0�5�<�A�D�@�<�0�#��m�h�[�W�Z�`�m�y���������������������y�mE�E�E�E�E�FF$F1FVFoF{F~FuFbFVF=FE�E�E�������������������
�����
�����������������������������	��"�"�	�������������������������	�����	������������������������������������
��
������������ؼ�������������!�$�'�'�����������ùìà×à�������)�6�>�9�0�)�����(�)�#��$�/�H�_�b�]�P�K�K�U�c�k�a�H�<�(ā�x�t�t�tāčĚĖčāāāāāāāāāā�û������������ûܻ����������ܻлúɺº����������ֺ����� �������ֺ��z�q�s�m�m�k�m�o�z�~�����������������z�z���������������Ľǽнݽ�ݽнĽ���������D{D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D{DzDtD{ > C R ` > f 3 k R > n C 1 P _ % : ` g � t N m 8 ` 8 V J  + , = , L 3 k / n 9 ) 8 E A g O M % 0 � b Q N j ; 2 @  t 6 � H 4 $ f y 0 ? < Z p N    D      �  )  *  ;  �  F  �  =  �  �  F  S  �  �  o  [  U  �  x  |  P  �    J  �  �  �  l  p  ]  j    �  �  |  8  c  �  Y  �  C  7  N  �  E  T  I  \  �  �  �  �  ~  �    2  �  �    �     �  _  U  1  ;  �  R=�t��D��;o���
���ͼ�C���`B��o�D����󶼃o��"ѽ�㼋C��8Q������P�ixռ��ͼ�9X��9X��9X����������+��/�m�h�t����P�D���D������#�
�T���Y���t��'����49X�T���q���D������<j�������
����m�h�����O߽]/�aG��ixսu��%������ٽ�t������u��+��{�$ݽ�����P��j�����9X��vɾ	7LBB��B�cA��_B��B�B��B@B�PB^�B ��BC�B#qB��B��B�cB��B�BεB?cB��B�tB�lBw�B�uB7yBݐBHB
��B
�B��B
�nB�B/�BgB	.�B�;B�QB!��B `�B�6B��B$�IBC�B5'gB��B�cBh%B-/�B��B	�BX�BA�BI�B`cB�B+$�B�BOBT�B &�B
vQB¿B��B	��B
��B('�B�ByB&�B]B=�BDB�A��IB��B?/B��BBgB?�B?B ±B�<B�B�pB�B�FBܣB�OB�1B>�B8�B�aB�BE�B��B=ZB :BIQB
�;B
�B=�B
�mB�BB�-BAuB	AcB�RB�B"�B }BBB��B$BB@�B5>tBōBÙB�%B,��B�BA�B@dB?6B?�B0�B�B+:B��BB?jB �B
�B�B@pB
��B-�B'��B��B|B&�7B3�C�F~B	wA��`A���A��@�juA�d�A�L]A�xA��QA��B�A�$�A�nAA�իAk5A�=�A��A�<�A��:@�A��MA�G�@ʽ�A�m�@�0@�m�Ai�LAz#=A�1(AO:VB8�A�A^njAC�=AXxAW�2AQ�?��IAC�]C�9�Atr�@�GTC���AJg�A��AC��AL��A,�A��EB	�YA�vA�,B��@��A��An$�C���A��A�\�A���A��,A]�A�}�A� CA���@��@>�A�IA%��C��~C�G�B�GA�'�A���A�|�@��A���A�t�A���A��-AĆ�B/AštA���A���Al�CA���A�s�A�~
A���@��KA���A��@�PMAǎq@�G@���Ao�Az�A���APkvBF;A�
�A] AD�AZ��AW!AO �?�%ACSC�7�As�@x�fC��AK ;Aމ�ADAL��A�A�B�B
AWA�&fA�z)B��@�/�A��Al�HC���A�TA���A�N�A�A�A�\Aϒ�AÄ�Aݣ�@��@<ǘA��A#m�C�Ͽ                                    `         !         *                        -          	   +         '            %      '            
   !          $                                  D            
      A   !         &         ,                                    5   #                                             )      )         +                  #                              !                           -      !            )   #                                                      +   #                                             )      '         +                  !                              !                           -                  )                     N9�7N�1M۲N_<�N�$�N�<kN5O:�O�[O�NA�P6"O�R�NcxO��N��OU��O9��O?M�Z�NI�INF��O��NE6)N��O"��N�P�BN���P
O'��O&��P(��NU�N��QO��>O!Z#N?�)O�)NM��N�1{O �lOa�!N�=N KsN���O��DOF��OX�N�	O��N�A.NK�IN�FN�c�O)I�O8	|P^O��O0�>N�?�N���OD;	O�	TO�^NIX�O�}WOQ��N�[4N�_�O�    R  =  �      �  �  c  �  �  �  �  +  �  F  �  �  �  P  6  �  }  1  �  �  �  �    �  �  /  �  :      �  9  �  �  �  L  �  
  E  %    p  Q      �  �  4  �  F  �  �  �  �  $  &  �  
b  �  }  �  �  �  �  c=��-<D��;�o%   ��o�o��o���
�ě��T���49X�+�D���T����9X��o��C����ͼ�t���t���t����㼛�㼣�
��1�+��9X��/��`B��P�C��C���w�\)�\)�t��49X��P�'���w�'�w�8Q�0 ŽixսH�9�T���@��L�ͽH�9�L�ͽL�ͽL�ͽL�ͽT���]/�]/�aG��ixսixսixս�%��7L��hs��\)���P��9X���罰 Ž�^5��������������������).5ABDCB5)���������������mmz|�����zwrommmmmmm	
#/3:94/#
				y��������������{~}y��������������������)3200.,)
@BNV[agkt{tg_[NB?99@7<>HUX^aaba_UH<:5577��������������������z���������������yotz#/<HUWazzn[UE<//##��������������������������������������������	
������������
#+,(#
������
#/@HJIH@</#
 
����������������������������������������)-66:6)'��������������������	)/58BGNSB5)'	�������������������������������ABJO[hmtwytqh[ONGBAA��������������������6Bhpss��zth[OB71206{������������{{{{{{{Zdft�������������g\Z��!)(������ktx��������������tpk�5B[_`\_XNB5	����RUYabhlkaZUSRRRRRRRR#'&$��5:7BNYgt������t[N=55��������������������������������������������������������������������������������������������������##/5<HRUXXUOH</###��
!*&+060,#
����
#&,*)#
����������������������������������������#/=HMMKC</#
knz���������znlechhk���������������()26BIOTTOOJB?6)"((����

��������������������������������������������������������������������������������������������������������������������������������#*/<N]gjomaUH<6.(" #')5BBFECB>5*))!''KO[hmopqqnnlkh`[WSLKtz���������}znttttttrt}�����������wtqorr����
	������������� )0*
������BNa���������tgTNED?B��������������������GPUbn{������{nb^UKGG��������������������Wanyz�������|zna[ZWW8<IJRUVUUSMIGDA=?<88<BEGEA<1//'##"! "#/<E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E��������������$�+�+�'�$�#���������������������������������������������������<�:�8�<�I�P�U�b�h�d�b�U�I�A�<�<�<�<�<�<ŹŹůŭũũŭŹ��������������źŹŹŹŹ�Y�M�M�L�M�Y�f�k�r���������������r�f�Y�#������!�#�/�3�/�)�#�#�#�#�#�#�#�#�ѿ˿ſƿѿݿ������#�&�������ݿ������������������������������������������Z�P�N�H�I�N�Z�g�n�s�v�����������s�g�Z�Z�H�H�C�A�H�S�U�Y�X�U�H�H�H�H�H�H�H�H�H�HƌƄƁƇƎƚƧ������$�)�-�"�����Ƴƚƌ�4�*�&�+�5�<�H�S�V�c�zÓÓÎÐÇ�w�U�H�4�b�\�a�b�n�{��{�u�n�b�b�b�b�b�b�b�b�b�b����������(�5�A�D�B�A�:�5�(����`�\�T�P�T�T�`�m�y�����������y�m�`�`�`�`ìâÚÖÔ×àìùÿ����������������ùì�����)�3�6�B�D�O�Q�P�M�H�B�;�6�)�������������������������������������������s�s�j�g�Z�X�Z�g�k�s��������s�s�s�s�s�s�M�K�@�=�>�@�K�M�Y�b�[�Y�S�Q�M�M�M�M�M�M�A�5�5�4�5�A�A�N�U�W�N�I�A�A�A�A�A�A�A�A��
��	���)�2�5�>�B�F�H�D�B�5�3�)�������'�4�:�@�4�'�����������a�_�]�a�d�l�n�q�z�|�|ÇÓÕÓÇ�z�n�a�a�x�t�l�h�k�t�x�������������������������x�����������������������������������������#����G�T�p�y�����������������m�T�;�#�Ŀ����������Ŀѿݿ��ݿܿѿĿĿĿĿĿ��t�_�Z�b�m�s�v¦²��������·²���������������ʾվ׾ݾ���׾Ҿʾ������b�Z�V�I�C�A�I�M�V�b�o�x�{�}�|�{�s�p�o�b�������������������*�C�L�P�N�C�6�*�����	�	��	��"�.�:�.�+�"���������f�e�^�Z�R�Z�f�s�����������������s�f�f���׾ϾԾ��������	�������	�����׾ѾѾ׾�����	������	�����׾;ʾ������������ľʾ׾߾���׾׾׾׺L�A�?�B�I�L�e�r�����������������~�e�Y�L�f�c�f�j�s�������s�f�f�f�f�f�f�f�f�f�fD�D�D�D�D�D�D�D�EEEED�D�D�D�D�D�D�D߿����������������������������������������!����!�:�F�S�\�_�e�l�s�u�k�_�S�F�-�!E7E2E5E7E@ECEPE\EiEpEiEcE\EXEPECE7E7E7E7����������������������������������������čąā�t�o�t�yāčĚĦĩĳĳĳĲĦĚčč�f�Z�O�H�J�M�Z�f�s�����������������s�f���������������������Ⱦ׾־ξʾ����������ּʼüɼӼ������#�'�����������T�R�H�D�>�;�8�;�H�T�U�a�j�m�s�t�m�a�T�T���������$�0�=�G�I�N�I�?�=�$���ŭŪŠŔŉŔŘŠŭŹ��������žŹŭŭŭŭ�I�H�I�M�U�b�m�n�{ń�{�n�b�U�I�I�I�I�I�I�O�O�C�A�C�J�O�\�h�h�u�v�u�h�h�\�O�O�O�O���������������������������������������������	�
���#�'�0�5�<�A�D�@�<�0�#��m�j�`�]�Y�\�`�m�y�������������������y�mE�E�E�E�E�FF$F1FVFoF{F~FuFbFVF=FE�E�E�������������������
�����
�������������������������������������	�������������������������	�����	������������������������������������
��
������������ؼ�����������!�"�%�%�!�����������ùìà×à�������)�6�>�9�0�)�����+�*�#� �%�/�<�H�X�`�\�O�J�J�U�_�U�H�<�+ā�x�t�t�tāčĚĖčāāāāāāāāāā�û������������ûܻ����������ܻлúɺ������������ɺֺ�����������ֺ��z�q�s�m�m�k�m�o�z�~�����������������z�z���������������Ľǽнݽ�ݽнĽ���������D{D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D{DzDtD{ > C R ` < f 3 k R $ n L 1 P : % 9 V g � t N m 8 ` $ V J  * , = * L 3 k * n / ) 8 F A E O @   � ` Q N j ; 2 @  t 6 z H 4 ' f u 0 ? ) Z p N    D      �  �  *  ;  �  F  )  =  o  �  F  J  �  �  �  [  U  �  x  |  P  �  `  J  �  �  ]  l  p  �  j    �  ]  |  �  c  �  4  �  �  7    ;  �  T    \  �  �  �  �  ~  }    2  �  �    �     �  _  U  �  ;  �  R  @9  @9  @9  @9  @9  @9  @9  @9  @9  @9  @9  @9  @9  @9  @9  @9  @9  @9  @9  @9  @9  @9  @9  @9  @9  @9  @9  @9  @9  @9  @9  @9  @9  @9  @9  @9  @9  @9  @9  @9  @9  @9  @9  @9  @9  @9  @9  @9  @9  @9  @9  @9  @9  @9  @9  @9  @9  @9  @9  @9  @9  @9  @9  @9  @9  @9  @9  @9  @9  @9  @9    �  �  �  �  �  �  �  �  �  �  �  ~  g  P  9  �  �  v  1  R  P  K  C  9  +        �  �  �  �  �  �  �  i  ;  
  �  =  7  2  ,  '  !            �  �  �  �  �  �  �  �  w  �  �  o  [  J  I  H  F  B  :  3  +  !    
  �  �  �  j  5  �  �      �  �  �  �  �  Y  !  �  �  V  
  �  c       �    
  �  �  �  �  �  <  n  e  W  E  0    �  �  >  �  �  -  �  �  �  �  �  �  �  �  �  �  �  �  �  ~  e  L  3      �  �  �  �  �  �  �  �  �  e  M  A  0      �  �  �  �  U  (  c  `  ]  Z  Q  F  ;  .         �  �  �  �  �  �  �  �  �  6  j    �  }  y  s  j  `  R  @  )    �  �  8  �  �    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �        #  .  E  �  �  �  �  �  �  �  �  �  �  �  J    �  a  �  P  �  m  �  �  }    t  h  b  f  f  b  U  E  .    �  �  �  w  6  �  +  ,  .  /  0  1  /  .  -  +  %        �  �  �  t  F    "  P  g  y  �  �  �  �  �  �  i  M  (  �  �  h      �    F  4  "    �  �  �  �  �  e  @    �  �  �  ]    �  �  �  �  �  �  �  �  �  �  �  �  �  �  m  E    �  �  |  @    G  �  =  �  �  �  �  �  �  o  O  (  �  �  �  q  %  �  3  A  \  �  �  �  �  �  �  �  �  �  �  �  �  v  e  X  N  E  ?  ;  7  P  Z  d  n  x  }  s  h  ^  S  H  <  0  $      �  �  �  �  6  *        �  �  �  �  �  �  �  ~  k  Y  6    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  y  }  p  `  K  9  (      �  �  �  �  �  D  �  �  T  �  y   �  1  1  -      �  �  �  �  �  �  u  P  $  �  �  �  �  n  K  �  r  d  V  F  2    	  �  �  �  �  �  �  �  �  o  ^  L  ;  6  �  �  �  �  �  �  �  x  1  �  �  D  �  �  ;  �  f  �  �  �  �  �  {  u  o  i  d  ^  Y  T  O  "  �  �  M  (    �  �  �  ~  t  k  a  R  4    �  �  �  �  �  �  H    �  v  :        �  �  �  �  �  �  �  �  �  p  Z  D  .    �  �  z  F  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  T  �  }  �  �  �  �  �  �  �  �  �  d  A    �  �  t  9  �  �  �  V    �  /  !    �  �  �  �  �  d  C  !  �  �  �    O  	  �  R   �  �  �  �  �  s  _  I    �  �  y  J  4  .    �  x  �  �  �  :  4  -  '  !        
      �    L  ~  �  �  �    ;    
  �  �  �  �  �  �  y  b  I  /    �  �  �  �  `    �    �  �  �  �  �  �  �  �  �  �  l  N  /    �  �  [  2  !  G  n  �  �  �  �  v  P  )  �  �  y  %  �  Q  �  �  �  �   �  9  .  #         �  �  �  �  �  �  �  q  ]  S  M  G  A  ;  �  �  �  �  �  s  [  <    �  �  ~  N  #  �  �  g    n  �  �  �  �  �  �  �  �  �  �  y  p  f  ]  Q  @  0      �  �  �  �  �  g  E     �  �  �  �  �  �  �  �  �  �  �  �  �  �  @  F  K  >  ?  8  '    �  �  �  w  H    �  �  s  J  M  X  �  �  �  �  �  �  o  ]  K  =  3  *  #        �  �  �  �  	m  	a  	�  	�  	�  	�  	�  	�  	�  	�  	u  	'  �  s    �  ,  �  ^    E  <  2  )            �   �   �   �   �   �   �   �   �   �   �   t  �  �  �  �  
      !  %  "        �  �  �  ]     �  >    �    	  �  �  �  �  �  y  I    �  �  G  �  �  -  }  �  6  O  _  g  o  l  k  m  j  _  N  7    �  �  E  �  �  w  �  Q  ;  $    �  �  �  �  d  R  V  O  A  .    $    �  �  �            �  �  �  �  �  �  h  0  �  �  T    �  I  d    �  �  �  X  (  �  �  �  �  �  �  S    �  b  �  |    �  �  �  �  �  �  �  �  �  �  �  r  \  G  1    �  �  �  [  )  �  �  �  �  |  h  U  A  ,    �  �  �  �  �  �  �  �  }  v  4  (      �  �  �  �  �  �  �  k  X  D  0      �  �  �  �  �    n  ^  K  @  G  L  L  G  >  4  *  "    �  �  �  P  F  F  D  >  5  )      �  �  �  �  �  j  L  &    �  �  �  �  �  �  �  �  �  �  �  y  c  G  "  �  �  �  \  )  �  �  p  �  �  F  
�  
    
�  
�  
v  
*  	�  	�  	/  �  $  r  �  �  �  4  �  �  �  �  �  `  9  
  �  �  d  *  �  �  {  >    �  �  Z  K  i  �  i  Q  U  >    �  �  �  �  �  �  [  4    �  �  �  $        �  �  �  �  �  �  �  �  �  �  z  i  Y  H  7  &  &  #           �  �  �  �  �  �  �  |  d  M  9  '  0  9  �  �  �  �  �  �  �  �  g  G    �  �  s  /  �  �  1  �  $  
b  
'  	�  	�  	p  	q  	B  	(  	�  	�  	F  �  �  9  �  5  y  n  J  !  �  �  �  �  �  }  \  8  $  �  �  v  E  "  �  �  U  �  C  z  }  p  d  W  J  =  .        �  �  �  �  �  �  �  q  [  F  �  �  �  �  �  �  �    l  V  =  !    �  �  s  '  �  s  J  c  o  {  �  �  �  n  F    �  �  e     �  �  (  �    -   �  �  ~  w  o  g  ^  U  M  F  A  =  9  0  "      �  �  �  �  �  �  �  �  p  ]  J  7  %    �  �  �  �  �  f  I  /     �  c  C  #  �  �  �  <  
�  
�  
O  
  	�  	Y  �  �    i  �    K