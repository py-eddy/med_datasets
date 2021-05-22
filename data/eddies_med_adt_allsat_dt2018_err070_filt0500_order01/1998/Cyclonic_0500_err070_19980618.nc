CDF       
      obs    F   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?�KƧ       �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       Mŏ:   max       Pyg�       �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ��^5   max       <#�
       �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?(�\)   max       @F�Q��     
�   �   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�z�G�    max       @v{��Q�     
�  +�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @)         max       @P�           �  6�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�         max       @���           7`   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �t�   max       ��o       8x   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A��   max       B3�*       9�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A��   max       B4u�       :�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       ?�N   max       C�ś       ;�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       ?��|   max       C��p       <�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          [       =�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          ;       ?   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          /       @    
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       Mŏ:   max       PIЎ       A8   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��$�/   max       ?��s�h       BP   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ��v�   max       <#�
       Ch   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?(�\)   max       @F�Q��     
�  D�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�z�G�    max       @v{��Q�     
�  Op   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @+         max       @P�           �  Z`   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�         max       @�@           Z�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         E$   max         E$       \   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�z�G�{   max       ?���     �  ]            P   /         	      	                     &               @         '                           $   !                        %                           *   0         &         [         /         	      	          N�!N��N>�Pyg�P:�MOP�O��N�'SN,�KN��O�#�N�QOf�O�H�N��N��O�׹N[�NF'UOdX)P[ P[6�O���N�n�P;��N;N�OF(3O�H@P_�O�xOM�N��;P�;Op��OMT�N?�NJ��O3Y�O#��N���N���PJ�O� ;N�
N9LRM�ɉNCi�Nj2xN,��Nz�O���P�CN�}ZO�@6P�HOÐ;OSo�O���O���N���P	��N��3O���O4A�O�1N��;M�~Mŏ:O�<#�
;o�D���o�o�t��t��#�
�#�
�D���e`B�e`B�e`B�u�u��o��C���C���C���C���t���t����㼣�
���
���
��9X�ě��ě�������������������/��`B��h�����o�+�+�+��w�'',1�0 Ž49X�8Q�8Q�8Q�@��@��H�9�H�9�L�ͽL�ͽT���]/�e`B�m�h�y�#�}󶽃o��o��7L���罩�罬1��^5�������������������
 !
���������gmz����������zaTH>?g�������$)!�������������������������������
#%-20'#
�������NN[gty�����}tg^[RNNN��������������������##/<>CG<1/(#"!######������������{{������Z[gt~tsg[XZZZZZZZZZZ#<HU^URPGB</#()6BHO[homhb[O6)!&(��������������������`aadmrxvqpmljfa^]]``�������
 ������-/<AHOH?<2/..-------TUafnz|}zzznka`UTTTT������	�����������
#Ii}������|U<0
!0IUbknxy{nUI<#P\bhu�����������nhYP|������������������|������
������������������������������)6663)26BO[htutmih[YOB60.2sx{~��������������tsRVcz����������zmaTQR�����������������������������������)5@BIJCB5)%)5BN[gt~~wg_YN)���

��������zz���������������zzGNP[d_[YONLBGGGGGGGG����������������������������������������]adntz����zywna]WWX]26=BENOOPPOBB<632222LOR[hinmjjh[XOLILLLL>BN[gt�������t[NF>>.104;HTafiheaTH;/,,.��������������������
  
[[]httuth[[[[[[[[[[[��������������������agtx������xtg^aaaaaa��������������������������������������������������������~|}������#(&�������������������������������������������������)5NYTPIFB5����,057<HNZZUNC</#
Z[g����������tg[XRRZ�
#/3@HJH</#
����W\]pt��������wsog`YW����������������������)>B<6)�����enz|��������znldeeee5BN[gnrutg[K5$qt{���������������vq��������������������NBB6)))36BOV[a[ONNNN./<>FC<;//..........��������������,/07;<BHJPTTMH</**,,���������������������������������������˼��������������üɼ����������������������@�7�3�/�3�>�@�B�L�M�L�H�@�@�@�@�@�@�@�@čćĉē��������6�B�0����*�*�����č�	���Ҿ����������ʾ��"�T�c�`�N�;�.�	�z�w�n�f�a�b�nÇÎÓàäìðìàÜÓÇ�zÆÀÃÇËÓàìù������������ùìàÓÆ�����������	������	�����������������������������������������������H�H�D�A�H�U�a�d�n�r�n�j�a�U�H�H�H�H�H�H����4�A�I�M�W�Z�Z�^�a�M�A�9�(� ��	��f�c�a�`�f�r�s�w�v�s�f�f�f�f�f�f�f�f�f�f�6�)�&�'�)�'�)�-�6�B�O�[�i�o�d�h�[�O�B�6�/�*�"��"�%�-�6�H�T�[�a�f�m�x�v�m�a�;�/�U�T�Q�H�C�9�3�<�B�H�U�a�f�i�i�c�a�[�]�U�������	���"�/�;�H�P�H�;�/�)�"��	�����O�G����������������*�6�M�\�g�f�\�O�N�F�K�N�Q�[�`�g�m�t�t�t�g�[�N�N�N�N�N�N�m�b�h�m�p�x�z�������������}�z�n�m�m�m�m�����������½Ľнݽ������ ���ݽнĽ������u�\�N�A�%�5�A�s����������������������������������о�4�A�L�N�P�N� �ݽн��������������������������ʾ۾���׾Ѿʾ���������x�v������������������������������|�`�;�'��.�;�T�Z�p�y�����Ŀ��ʿ���������� ����������������������������������������������������������������"�������"�.�;�G�I�S�T�W�P�G�;�.�"�	�����׾������������ʾ׾�������	��ƳƚƁ�p�h�c�h�uƁƎƚ�������������������������������$�0�6�7�2�0�$�������������������.�7�;�G�O�M�G�;�.�"�������������������������������������������������������� ��C�O�f�h�h�d�J�?�*�����z�z�|������������ʾ޾Ҿʾ��������������������������������������	����	���I�=�=�8�=�I�S�V�b�c�b�V�I�I�I�I�I�I�I�IǔǊǈǃǈǔǡǦǭǡǔǔǔǔǔǔǔǔǔǔ�g�Z�N�5�(�����!�(�5�A�N�Z�f�m�s�g�g�g�]�Z�M�N�P�Z�g�s�������������������s�g�S�I�F�D�F�S�W�_�l�x���z�x�l�d�_�S�S�S�S�������������������ûƻŻû�������������������������������������*�,� �������s�g�Z�N�D�9�3�/�5�A�g�����������������s�����������������������������������������$���$�0�=�@�=�7�0�$�$�$�$�$�$�$�$�$�$�'�!�����'�'�*�*�'�'�'�'�'�'�'�'�'�'������������������������������������������������������������������������������������������������������������������L�I�J�L�P�X�X�Y�`�e�g�p�h�e�Y�P�L�L�L�L�������������)�B�F�Q�W�U�O�H�B�6�)����������������!�.�8�9�2�&����ּ������������� �$�$�%�$������������������A�5�*�-�5�A�Z�g�s�������������s�g�Z�N�AÇ�n�a�U�P�I�D�H�U�zÓìù������ùìÓÇ�����������������ûܻ������
���лû��{�y�q�s�u�{ŇŐŔŖŠŨűűũŤŠŔŇ�{EEE
EEEE7ECEPE]EjElEiEbEXEPE7E*EE¿²¦¦²¿�����
��
�����¿��������������!�-�.�8�-�!�������U�A�:�3�:�>�`�{���Ľнֽܽ޽ѽ������l�U�����������������ĿɿѿԿӿҿѿĿ�������ĚėėĠğğġĪĿ��������������ĿĳĦĚ��������������������
����#�)�1�!�
�����z�|�����������»Ļлֻлͻû�����������������úù÷ììæêìîùÿ����������FFFFF$F1F4F1F&F$FFFFFFFFFF���	��	��"�,�'�"����������D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D� ( 8 ` S T .  ] G 3 ? c C . ] � d P � ( T P u R v ; G 9 L T 9 H + # A + g G [ E \ - I c D B H i t j s = p G K D 4 , 9 X F T 9 x ; j n K � D  4  /  X  ^  �  �    1  A  �  :  S    5  \  ,  ^  ~  �  �  F    �  �  �  K  C  �  e  �  	  �    �  �  �  V  Q  �  h  �  �  �  :  6  P  *  �  �  �  �  .  ?  �  �  �  �  �  �  �  !    �    �  V  �     !  0��o��o�o��9X�aG���`B��󶼛�㼃o���
�\)��o�\)���\)���
�aG���9X��1�\)�D�����T�+��`B�q����j���ͽ\)�#�
�@��H�9�8Q�o�}�u�'t��'m�h�49X�8Q�T����������49X�D���H�9�@��L�ͽ�+�H�9��E�����aG����P��-��hs��7L�t����
��O߽�"ѽ����Q콓t���-��^5�� Ž�-���mB�<B$��BdB L&B~�B3�B�WB	^�B�,BP�B��B	OB/YBZ&B �TA�ѹB�B�By�B"��B'�B&�|B3�*B�aB,/FB*9B*B��B��B *�B�Bl�BB٩B��BgfB�"BdB�_B�KB8�BX�B	�XA��B��BB��B33B	��B8B!>"B=�B-�hB߰BީB�B%�qB	�LBvaB	�fB!�BeFBYB�B
�BB��BJ�B�xB9�B��B��B$�B=A��eB�
B;�B:
B	�Bo#BBB�,B	P	BMB"~B �SA��B̓B��B��B"CsB&�SB&��B4u�B��B,?nB*�B�B@�B�tA��^B=�B�
B��BA�B��B��B?�B�B�RB�|B<B@_B	�
A��B�OB6FB��BI�B	��B�B!:TB?B./:BLxB@�B:�B%Q�B	�lBA�B	��B?�B��B;�B�B�B�yB@4B��B4B��A��,@�Ȇ?��JA���AY��A�I�A�R$AY2�A�9Aŵ$A80BAA��AؾkA� ^A��A��dA���A�['A���A*&�A���A, �AQ��@�l�Ai��A��tA�)�AaF�AS�WB�B��A^��As��A�ØAIGA��*BteBeA���A���@��@���A���A�ёA�02B
J?�NA�{A�J�Bd?�HA�WrA-B	�A�a�Aɓ�@�]dA�tC���A�%�@`d1ARAx9	A�A�J@�`�A��C�śA��mC��A��z@�܉?��|A�{�AT�8A�s�A��AYrXA�|�A�eA7��AA��Aב�A�g�A�dA��NA��A�~ A�� A*�CA��<A)�^AOǩ@�::Ad�:A�`�A��AaWoAT��B�-B	�A_RAs!6A���AH��A�}'B��B@ A���A�@�@�p�@���A���A���A�(�B
s?��|A���A�OB?�?���A֒A��B	*A���A�w#@�_�A�{.C���A��Q@[ĞA	�Ax�VA↨A��@��Ȧ<C��pA�jOC�	�            Q   0         
      	                     '               @      	   (                           $   !                        %                           +   0         &         [         /         	      	         !            ;   3                                    #            9   3         7               )            '                           )   #                           -      !   '   %         #      +      !                              /   /                                                /   '                        )            !                              #                           -      !   '   !         !      '                        N�!N��N>�P$�2P#))N�#6O���N�'SN,�KNW�<O�KN�QORxqO]зN��N��O�NǖNF'UO7SaPIЎO��#O���N��:Okp�N;N�OF(3O�%eP_�OD�)O
�N��;O��N���Nĵ�N?�NJ��O3Y�O#��NG�N���O��O� ;N�
N9LRM�ɉNCi�Nj2xN,��Nz�OY�O�4�N�}ZO�@6P	_O��9OSo�O tO��6N���O��N��3O��O4A�O�1N��;M�~Mŏ:N���  �  P  E  e  �  �  s  �  5  A  �  #  �    W  �  �  ~  �  �  �  �  F  B  �  �  �  �  �    ;  �  �  �  ,  �  �  �      �  �  .  p  �    �  �  �      J  �  _  c  �  �  �    �  H  �  �  �        }  #  	"<#�
;o�D�����ͼD����C��#�
�#�
�#�
�T����1�e`B�u��t���1��o�ě���t���C����
���
�#�
���㼬1��㼣�
��9X�ě����ͼ������������C���P�C������o�+�\)�+�L�ͽ'',1�0 Ž49X�8Q�8Q�8Q�Y��D���H�9�H�9�P�`�T���T���� Žm�h�m�h��C��}󶽇+��o��7L���罩�罬1��v��������������������
 !
���������Ymz�����������aTLIKY����&
����������������������������������
#$,1/&#
������NN[gty�����}tg^[RNNN��������������������"#&/<<A=</##""""""""��������������������Z[gt~tsg[XZZZZZZZZZZ#<FHQOFB</*#).6BO[hljh][GB6)(#&)��������������������`aadmrxvqpmljfa^]]``�����
��������./<=HNH><1/.........TUafnz|}zzznka`UTTTT������������������#0I{�����{UI<0)4<IU[ceiigbUI<1,$%)P\bhu�����������nhYP������������������������������������������������������������)6663)26BO[htutmih[YOB60.2���������������yz|RVcz����������zmaTQR��������������������������� �������)5@BIJCB5)%)BN[^hmqrng[NB&��������������������������������GNP[d_[YONLBGGGGGGGG����������������������������������������]adntz����zywna]WWX]36BBCLNNB>6433333333LOR[hinmjjh[XOLILLLLKO[gt��������tg[UPKK.104;HTafiheaTH;/,,.��������������������
  
[[]httuth[[[[[[[[[[[��������������������agtx������xtg^aaaaaa���������������������������������������������������������~����#'%���������������������������������������������������)5NSRPHEA5���#0<DKWVSIE<0(!Z[g����������tg[XRRZ 
#,/28:/-#
	   X^`ht����������tgb[X����������������������)5;90)�������enz|��������znldeeee5BN[bgmooi[B5&qt{���������������vq��������������������NBB6)))36BOV[a[ONNNN./<>FC<;//..........��������������,/18<<<HIOSSLH</++,,���������������������������������������˼��������������üɼ����������������������@�7�3�/�3�>�@�B�L�M�L�H�@�@�@�@�@�@�@�@ĿĚĖğĳ���������
����
���
����Ŀ��ľ��������ʾ���	�"�.�T�\�Y�;�.��	���z�o�n�l�n�q�zÇÓÔÙÜÓÇ�z�z�z�z�z�zÓÇÁÄÈÑÓàìù������������ùìàÓ�����������	������	�����������������������������������������������U�L�H�E�G�H�U�a�a�m�b�a�U�U�U�U�U�U�U�U�(��������(�4�6�A�M�M�R�N�M�A�4�(�f�c�a�`�f�r�s�w�v�s�f�f�f�f�f�f�f�f�f�f�6�/�)�'�'�)�*�6�B�O�[�h�m�h�b�f�[�O�B�6�4�/�*�(�-�6�>�H�T�W�a�b�m�o�r�n�m�a�H�4�H�G�>�=�H�S�U�W�`�a�d�a�\�U�H�H�H�H�H�H�������	���"�/�;�H�P�H�;�/�)�"��	�������������������*�6�<�H�O�O�C�6�*��N�K�N�N�S�[�_�g�k�g�e�[�N�N�N�N�N�N�N�N�m�b�h�m�p�x�z�������������}�z�n�m�m�m�m�Ľ����������Ľнݽ������ ������ݽн����x�m�_�G�D�g�s���������������������������������������Ľ���(�4�<�=�-��ݽ����������������������������ʾ۾���׾Ѿʾ�������z�|������������������������������`�T�@�4�/�-�1�;�G�O�T�g�m���������y�m�`������� ����������������������������������������������������������������"�������"�.�;�G�I�S�T�W�P�G�;�.�"��׾������������׾���������	�������ƳƚƁ�p�h�c�h�uƁƎƚ�������������������������������������$�0�3�4�0�$�����"��	��������	���"�&�.�;�@�G�;�:�.�"���������������������������������������������������*�6�C�O�Y�\�\�U�C�9�*��������~�����������������������������������������������������	��
�	�����������I�=�=�8�=�I�S�V�b�c�b�V�I�I�I�I�I�I�I�IǔǊǈǃǈǔǡǦǭǡǔǔǔǔǔǔǔǔǔǔ�g�Z�N�5�(�����!�(�5�A�N�Z�f�m�s�g�g�g�]�Z�M�N�P�Z�g�s�������������������s�g�S�L�F�F�F�S�_�l�v�l�b�_�S�S�S�S�S�S�S�S�������������������ûƻŻû���������������������������������������������s�g�Z�N�D�9�3�/�5�A�g�����������������s�����������������������������������������$���$�0�=�@�=�7�0�$�$�$�$�$�$�$�$�$�$�'�!�����'�'�*�*�'�'�'�'�'�'�'�'�'�'������������������������������������������������������������������������������������������������������������������L�I�J�L�P�X�X�Y�`�e�g�p�h�e�Y�P�L�L�L�L�)�����������)�8�B�L�O�S�Q�O�B�6�)�������������!�.�7�8�1�%����ּʼ������������� �$�$�%�$������������������A�5�*�-�5�A�Z�g�s�������������s�g�Z�N�AÓÇ�n�a�V�Q�J�G�H�a�zÓìùÿ����ùìÓ�������������ûлܻ���������ܻлû��{�y�q�s�u�{ŇŐŔŖŠŨűűũŤŠŔŇ�{E*EEE!E*E1E7ECEPE\E\E^E\EZEPEPECE7E*E*¿²¦¦²¿���������
������¿��������������!�-�.�8�-�!�������_�L�C�I�`�������Ľнսӽ˽��������y�l�_�����������������ĿɿѿԿӿҿѿĿ�������ęĘġĠĠģĦĭĿ������������ĿĹĳĦę��������������������
����#�)�1�!�
�����z�|�����������»Ļлֻлͻû�����������������úù÷ììæêìîùÿ����������FFFFF$F1F4F1F&F$FFFFFFFFFF���	��	��"�,�'�"����������D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D� ( 8 ` O V   ] G - ) c . 4 > � E @ � ! N _ u < s ; G 9 F T 2 : + 6  ( g G [ E E - 8 c D B H i t j s 6 m G K ? 0 , 5 R F S 9 v ; j n K � @  4  /  X  3  4  �  �  1  A  `  [  S  �  �  �  ,  7  C  �  �  �  �  �  �  �  K  C  �  %  �  �  ?    �    �  V  Q  �  h  b  �  6  :  6  P  *  �  �  �  �  �    �  �  }  i  �  \  i  !  S  �  �  �  V  �     !    E$  E$  E$  E$  E$  E$  E$  E$  E$  E$  E$  E$  E$  E$  E$  E$  E$  E$  E$  E$  E$  E$  E$  E$  E$  E$  E$  E$  E$  E$  E$  E$  E$  E$  E$  E$  E$  E$  E$  E$  E$  E$  E$  E$  E$  E$  E$  E$  E$  E$  E$  E$  E$  E$  E$  E$  E$  E$  E$  E$  E$  E$  E$  E$  E$  E$  E$  E$  E$  E$  �  �  �  �  �  �  �  n  I  '  �  �  �  A  �  �  r  )  �  �  P  J  C  <  6  /  (  "        	    �  �  �  �  �  �  �  E  ?  :  4  ,        �  �  �  �  �  u  \  C  )     �   �  �    6  U  e  a  N  .    X  S  /  �  k  �  $  x  �     �  �  �  �  �  �  �  y  I  *    �  �  N  �  �  �  �  �  �   �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  y  Q    �  B  �  n  r  i  Y  H  4      �  �  �  �  �  Z  (  �  �  E  .  *  �  �  �  �  r  d  T  C  1      �  �  �  g  ;    �  �  �  5  /  )  $             	                         (  :  J  W  a  a  `  Q  ?  +    �  �  �  �  �  o  O  /  w  �  �  �  �  �  �  �  �  �  �  p  P  -    �  �  �  r  �  #  $  %  &  '  (  *  +  ,  -  -  -  -  ,  ,  ,  +  +  +  *  �  �  �  �  �  �  �  �  �  �  m  (  �  �  y  K  B  L  #  �                �  �  �  �  x  Y  :       �  �  f  )  �  �  #  9  E  U  U  H  /    �  �  �  V    �  �  r  �  �  �  �  �  �  �  �  �  �  }  v  o  g  _  W  O  H  B  ;  5  .  �  �  �  �  �  �  �  �  �  a     �  �  9  �  i  �  8  �  ,  v  x  z  |  �  �  �  �  �  �  �  ~  l  X  D  0  �  �  _    �  �  �  �    v  k  `  U  J  <  *       �   �   �   �   �   �  �  �  �  �  �  �  �  w  [  <    �  �  �  H    �  �  P  !  �  �  �  �  �  |  S  $  �  �  �  ~  T    �  �  �  e     y  N  d  `  V  k  |  �  �  |  k  M    �  l    �    ]  ]    F  <  0  #      �  �  �  �  x  Y  A  (    
  �  �  q  ?  .  7  ?  A  ?  =  7  1  )         �  �  �  �  �         �  �  �  r  b  a  \  �  ~  �  y  ?  O  .  �  �  [  �  �   d  �  �  �  �  �  �  �  �  �  �  �  �  }  t  g  [  O  B  6  *  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �    |  �  �  �  �    j  T  =  #  
  �  �  �  �  �  �  g  ?     �  �  �  �  �  �  �  �    l  V  ?  $    �  �  ^  *     �   �      �  �  �  �  �  �  �  �  q  P  (  �  �  �  K  �  �  S       1  :  :  /      �  �  �  Y  $  �  �  h    �  4  �  �  �  �  �  �  �  �  �  �  �  �  ~  U  #  �  �  �  S    �  �  �  �    m  Z  F  1      �  �  �  �  �  �  |  m  _  P  �  �  �  �  �  �  �  �  �  �  �  �  �  c  '  �      F  4  �  �  �  �    '  +      �  �  �  �  m  '  �  4  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  h  1  �  �  O     �  �  �  �  �  �  �  q  ^  L  ;  &    �  �  �  �  j  ;    �  �  �  �  �  �  l  P  3    �  �  �  �  m  H  $    �  �  �      �  �  �  �  �  �  �  �  �  �  g  I  (  �  �  Y    �        �  �  �  �  �  �  �  �  �    ]  9    �  �  l  (  o  y  �  �  �  q  Y  @  &    �  �  }  ;  �  �  '  �  Y  �  �  �  �  �  i  L  .    �  �  �  �  ^  H  *  �  �  q  '  �  �  �  �      (  -  (      �  �  �  �  I  �  �    �    p  Y  <    �  �  �  �  ]  7    �  �  �  s  <  �  �  i  E  �  �  �  �  �  �  �  �  �  �  �  �  �  �  z  k  \  M  >  /      �  �  �  �  �  �  �  �  �  o  ]  G  ,    �  �  �  s  �  �  �  �  �  �  �  �  �  �  u  `  L  3    �  �  �  �  d  �  �  �  �  �  �  �  �  �  ~  r  e  Y  O  K  H  D  A  =  :  �  �  �  �  �  �  �  �  �  �  �  �  y  _  E  +  6  J  ^  r    �  (    �  �  �  �  U  (  �  �  �  b  ,  �  �  }  ?  �        �  �  �  �  �  �  �  �  �  �  �  v  Y  6     �   �  /  <  H  H  ;  '    �  �  �  O    �  �  &  �    *  �  Q  �  �  �  �  �  u  `  A    �  �  �  ;  �  �  e    l  �  �  _  [  X  U  N  F  >  A  H  P  P  H  @  8  0  '  "  ,  7  A  c  M  6    �  �  �  �  �  �  �  �  �  �  �  �  k    �  p  �  �  �  �  �  y  `  G  .    �  �  �  y  +  �  m  �  \  �  �  �  �  �  �    k  P  /  
  �  �  �  �  �  k  G  4  V  �  �  �  �  r  X  <  &       �  �  �  �  �  �  �  �  z  H  �  �  G  �  &  �  �      �  �  �  n  %  �    C  	�  8  q  .  U  �  �  �  x  Z  6    �  �  o  7     �  �  �  �  �  `  $  H  >  2  !  
  �  �  �  �  {  \  F  3    �  �  �  o  <  	  �  �  �  �  �  �  �  �  �  x  H    �  �  7  �  D  �  �  �  �  �  �  �  �  �  �  s  ]  F  -    �  �  �  ^  	  �  J   �  �  �  �  �  {  H    �  �  8  �  v    �  A  �  o  �          �  �  �  �  �  �  r  Y  @  '    �  �  �  �  l     �      �  �  �  o  C    !  <    �  �  �  I     �  D  �  t    �  �  W  .    �  �  �  �  �  -  c  V  I  ;  +    
  �  }  |  {  z  y  x  w  f  N  5      �  �  �  n  >    �  �  #                 �  �  �  �  �  �  �    f  M  4    	  	   	  �  �  �  �  d  2  �  �  U  �  U  �    c  �  �  /