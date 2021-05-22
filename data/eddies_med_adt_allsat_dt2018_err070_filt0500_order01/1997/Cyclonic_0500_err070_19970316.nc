CDF       
      obs    P   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?��+I�     @  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M�RE   max       P�8W     @  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ���   max       =L��     @   ,   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?
=p��
   max       @F'�z�H     �  !l   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��p��
<    max       @v|Q��     �  -�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @+         max       @P@           �  :l   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�K        max       @���         @  ;   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �+   max       =��     @  <L   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�4   max       B/ۗ     @  =�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A���   max       B/�     @  >�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >U��   max       C���     @  @   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >?b!   max       C���     @  AL   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          X     @  B�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          A     @  C�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          A     @  E   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M�RE   max       P�8W     @  FL   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�bM���   max       ?�!-w1�     @  G�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ���   max       =L��     @  H�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?
=p��
   max       @F�����     �  J   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       �ə����    max       @v|Q��     �  V�   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @)         max       @P@           �  c   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�K        max       @���         @  c�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         CY   max         CY     @  d�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?t��n   max       ?��ߤ?�        f,                              #   '      	      &                     @               '   '                  ,   	      	         &      =               8                  (         "      	   #         *      	   X   
            P   /                           	N�TO�G�O*MN��N���O%��OM[xNp�\N�^O�4�OA��OF�N���O�ژO�p�Nw��O+UkO�>�Oj �N��rN�B�P�8WOxoO��aN]k\O�GTP&i�P��]O��nN�CNh�}O��O�\PYK�OLNSNV),N��O%�|O�j�O�6PP�hN0&Oh[VO̋N���PR��Nƾ�NZ�N�X�N1v�M��VO}��N��O�W�O�ݕOB6pNR��O��N+��O�[�PC�Ox�9N��O`v�N]߾NH�M�REN�L�P'.O��^N��\O�y�Ot *O1z N �2N���N��Nt	NZ�=L��<��
<���<49X;��
;D��;D��;D��;D��:�o%   ��o��o�o��o���
���
���
�ě���`B�49X�49X�D���T���u�u�u��o��t���t����㼛�㼛��ě����ͼ���������/��`B��`B��`B�������o�o�o�+�+�+�C��\)�t�����w�#�
�0 Ž0 Ž0 Ž49X�49X�8Q�8Q�8Q�8Q�8Q�@��@��H�9�L�ͽP�`�T���T���T���e`B�u�u�����1�����������������������������������������������������������������������������������#)*+)'	��''(��������/5@DN[frvtog[NB97:5/#/1<B<5/+#��������������������#/6;>>8/#
��)6BHOS[chjh[OB5))$)5BNNQPOMLHB5+)%""$ )+2)%)3BFKMPOQNB5)"#%)6BJX\g[OB6)"  "/;DHPMH<;1/-%"    ��������������������%*5BO[hrt���}~}th60%NV[_gtw|�����g[NFFGN 
!!#$#
      wz�������zrtstwwwwww#0U{������{[<0
��gnqz���������znidggBFO[hqswta[SB:<7<<?B������������������������������������������#5;>=@?580
�����GSm�������������rOIG��������������������xz|�����������zzxxxx�����������������������������������������*6CIOJ61*�����)Ohv����{o[OB5*��)6;B96*) ����������������"�������������������������������^ajmz��������zylda`^��������������������R[ht{xtqh[[TPPRRRR��������������������#/3<=<1/%#".47862)����������������������������������������������-3+�������� #)-/:<?>><<1/,#  ���������������������������������������������������������������������������������������������������������������������������������������������������������{������ynaXUMMQUZan{()25BEBBA5)(((((((((����������������������������������������PUahnz��������znaVQPQ[htpt���������o^KHQt�����������tmijhjkt��� �����������������
 &' 
�����������������������������������������������ghhtwuthd`ggggggggggU[[dgmt������tttg[UU������������������Ngt}�������tg[NHDELN./7<HKONHH<4/.......OUan������zvnaZUPNLO#0<IUXWI<80#
	NOTgt�����ztg[OMNONvz������zyvvvvvvvvvv��������������������!#0<>CHC<20,(#!!!!!!����������������������������������������E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�������������%�/�9�<�=�A�J�O�W�P�<�/��ȽĽ����������Ľнݽ�������������ݽȾ4�2�(�"����(�4�A�M�Z�[�[�Z�S�M�A�4�4�ֺϺϺֺߺ������������ֺֺֺֺֺ־�w�t�u�w��������������������������s�q�f�Z�N�N�Z�f�s�x���������������x�s�t�s�q�s�t�}�t�t�t�t�t�t�U�Q�P�O�U�`�a�e�c�a�U�U�U�U�U�U�U�U�U�U��ŽŽ���������������"����������Ƽf�b�^�[�Y�X�Y�]�f�r�������������~�r�f��������������������*�0�4�*�&������L�F�@�9�=�@�I�L�Y�[�b�d�Y�Y�L�L�L�L�L�L�(������������A�C�F�K�K�K�A�5�(�F�:�-�!��#�%�-�F�_�����������|�x�l�S�F�����	��	���"�%�/�0�/�"������������������	��"�/�;�>�<�;�"��� ������������������¿¹´¿������������
������������������������������������������ìçàÓÈÓàìðùü����ùìììììì�g�c�e�g�s�����������������s�g�g�g�g�g�g���m�Z�P�;���5�N�����������������������������������������������������������޿�	�������	��"�;�m�~���������m�T�;�.����������������� ���������/�"�������/�;�E�T�W�b�_�Z�T�H�;�/�Z�R�M�P�g���������������������������s�Z�Z�5���A�Z�p�s�����������������������ZƎƁ�u�j�h�h�h�jƁƎƧƯ����������ƱƧƎ�M�E�A�4�0�0�4�?�A�E�M�Z�Z�_�^�Z�M�M�M�M�)�����)�6�A�B�C�D�C�B�6�)�)�)�)�)�)���y�m�`�\�U�Q�T�m���������ƿͿʿ��������G�.�	�������	�"�G�`�n�{���y�m�`�T�G���t�q�z�q�N�G�Z�s������� �	� �����������T�P�N�N�H�A�?�H�T�a�f�m�m�o�o�n�m�i�a�T�[�[�W�[�g�h�tĀ�v�t�h�^�[�[�[�[�[�[�[�[�����������ÿĿʿʿĿ��������������������U�a�n�q�x�z�|�z�n�k�a�U�H�G�A�G�H�T�U�U����������������������������������������������������$�1�6�6�2�$����ݽֽֽսݽ����	���������ݽݽݽݿ���ھ׾�	�"�.�T�`�y�������y�`�G�.��6�3�4�6�@�B�E�O�P�O�N�B�6�6�6�6�6�6�6�6�@�5�A�E�M�Z�f�s��������������s�Z�M�A�@���������������������������������������������'�*�4�@�M�U�M�G�@�;�4�'�����f�Z�c���������!�.�9�6�!�����ּ����f�<�6�/�#��#�&�/�<�H�U�U�a�b�a�]�U�H�<�<�@�?�6�4�1�4�@�G�M�Y�[�Y�X�M�@�@�@�@�@�@���������������������žʾооʾ���������ìæàÜÚØàáìîùïìììììììì�O�L�G�O�\�h�j�h�b�\�O�O�O�O�O�O�O�O�O�O�x�l�`�Z�[�_�l�x����������������������x�b�V�U�O�U�\�b�n�u�{ńŀ�{�{�p�n�b�b�b�b�~�j�h�o�~�������ɺֺ����ֺĺ������~�T�;�.�(�#�$�.�;�G�`�t�y�~�������y�m�`�T�������ĿϿѿӿݿ��ݿؿѿĿ����������������������������������������������������t�h�O�E�@�E�[�tčĦĳĿĿĳĚēĕčā�tƧƞƧƨƳ��������ƳƧƧƧƧƧƧƧƧƧƧ�;�7�.�"����$�3�;�H�T�\�^�^�a�_�T�H�;�0�'�#�<�I�Y�I�<�P�{ŠŹ������ŹŠ�{�b�0ķĸĿ�����������
������
��������ķD�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�EED�EEEEE*E7ECEPEWE^EaE\EWEPECE*E�ɺº������Ⱥɺʺֺ�����ֺɺɺɺɺɺ��������������������������	����'�'�(�'�����������������¿¹²§¦²¿�������������������ؼY�/�"�%�%���'�B�M�f�k�g�p���������Y�����
��#�/�<�H�V�[�]�e�j�d�U�/�#��FFFFFFFF$F1F1F9F1F-F$FFFFFF�������������ùܹ��������ܹӹù������лû������������ûлػ������������������������������� �&�&�)�)�'�����������������������������������������������������������
���
�������������������������� ���������������������������ûлܻػлû��������������������!��!�$�.�0�9�:�=�D�F�:�6�.�!�!�!�!�!�! @ > " ;  . 9 b � 0 J < : E = i Y i B O P Q N g M 5 L X : , D 8 ^ ` u ` U * U / % > G _ 5 V p S V G h R 4 X M  N R 8 L 6 R I V < 9 d > L R % ' : 9 7 ` D Y $ h  �  ]  d    �  g  �  �  �  0  �  �  �  u  �  �  �  �  �  �  �  V  N  .  i  9    z  2  �  �  �  �  �  �  _  �  �  �  =  #  �  T  -  B  �  �  �  v  �  �      �  �  �  �  \  �  N    v    A  �  l  [      �  �  �  B  �  �  4  �  �  t  �=��o��o<o�#�
��o�o�t��o�C���㼓t��t��T���,1��`B���
�u��j�T�����㽗�P��j��P���
��`B�Y��]/��h��j�ě��t��,1��7L�+�+�C��H�9�,1�����㽸Q��P�H�9��P��P��-�0 Ž49X�#�
�'�㽛��8Q�}󶽕���%�T�����-�@��e`B�� Ž�%�Y��+�aG��L�ͽY��e`B���Ƨ�%���P��7L�����%�����\)��Q���B�BY�B*OB�B�BK9B��B�xBpBd�B��Bi\B��BB)�A�4B�<B(/B	RBB9�BfZB&�8B��B�B��B�B�sA��B�
B�DB�"B+-B/ۗB�AB�3B8�B�B!v�A��BW�B�!B�ZB?B�B*%B)��B-zCB$�BZ�B!AB�B�[B jBv�B�%B�?B�BU�B�B^�BSnB
AtB
5�B��B�B#(mB1YB�B	��B �B	��B�B�4B%aB	b�B :B�SB%�IB�>B��B;B@�B)�~B|�B��B?�B�)B�B��B�B��B3�BB�BԹB?�A���B�B��BՉB?�B��B&��B<�B9B��B2�BU9A���B�B�2B�IB*��B/�B&7B��B@XB�B!��A��MB��B�lB�wB9�B�B	B)�XB->|B�B>.B!9�B�MB��B @�BP�B�=BxhB@�B@>B�B�(BRmB0�B
?�B9�B��B#=-BmB��B	��B �B	��B��BIuB%EtB	M�A���B �B&:�BR~B��C�1�A��JA)��A:TR@H�nAG��ABVgA��NA�+8A�Ш@�#A���?�-wA�k]@�}A��1A�V�A���A�¼A�a�A���A�T.A���Ae 3A�	�A�(A�7�A�B#�A<-�A��Ap3�A\�	A��)A�}�AۧAw1�A�D�B�B	�A.5:Ac"/A�O.ADA���@���@�GAĈH@��<AMF�A��B�p@��A�6�@�Ag<�Ay�A���A�1�B;A��UA��A�#�C�@C���@9�B��?�LYA�/0@� LA�4zC���>U��@��A���A�LPA��3A1F�@�u�A�C�8�A�u(A*A:ι@K�AG�AB�uA��A�}GA�t@��zA�(?̒1A��@���A�� A���A�ˉA��4Ǎ	A���A�"BA�ayAbkA���A�ysA��A���B�A;�+A�RAm��A[�
A��vA�!A�~�Av�Aƃ�B?�B	?XA/W8Ad�A؀AD�>A�i@� PA�Aħ@�߽AM��A�s�BW�@��A��f@TyAh+eAz��A���A���BV�A��lA��OA��C�C��s@4N�B�?�kA��@���A�	,C���>?b!@�ܥA���A�`A�TA2��@��A�                              $   '      	      '                  	   @               '   '                  ,   	      	         &      =               8                  )         "      
   #         *      	   X               Q   /                           
                                                                  A      %         -   =            !   -   ;                        1               9                        !            %         7                        +                                                                                                A      %         '               !      9                        !               9                                             7                                                      N�TO���N��RN��NL��N�kTNJ�NG��N�^O�4�N��O�:N���O��@O,Q�Nw��O+UkO�>�Oj �NP��N�B�P�8WN�JO��aN]k\Ow��P(eO���O��nN�CNh�}O��N�"�PK�EOLNSNV),N��O%�|O\��O�6O�L�N0&O-A~O̋N���PR��N9INZ�N�X�N1v�M��VOQ;_N��O���O�6�OB6pNR��O�bN+��O&PC�Oe�N��O �N]߾NH�M�REN�L�O�`Og;�N��\Ou��N� �OZ,N �2N���N��Nt	NZ�  �    �  �  �    n  �  i  �  W  �  N  [  ~  @  Z  8    �  �  �    �  �  �    �  �  �  ;  �  G  y  P  �  �  g  �  2  �  '  �  2  H  �  �  �    �  �    "  �    �  �  �  �    h  �  �  J  *  �  �  2  �  	�  a  �  Q  �  j  �  [    �  �=L��<���<e`B<49X:�o���
�D��;o;D��:�o��o��o��o�D����o���
���
���
�ě��o�49X�49X�e`B�T���u��C�����C���t���t����㼛����������ͼ���������/��`B�C���`B�]/���C��o�o�o��P�+�+�C��\)�'��#�
�,1�0 Ž0 ŽP�`�49X�D���8Q�<j�8Q�y�#�8Q�@��@��H�9��{����T���]/�m�h�ixսu�u�����1�����������������������������������������������������������������������������������&(!�������������LN[ghkg\[UNILLLLLLLL#/0<?<1/.#��������������������#/6;>>8/#
��16>BOX[_`[[OFB651111()5BKMLJHCB50))&$%(( )+2)&)5BEJKONNEB5)(/6BEORUWVOFB@63*()( "/;DHPMH<;1/-%"    ��������������������%*5BO[hrt���}~}th60%NV[_gtw|�����g[NFFGN
#$#
wz�������zrtstwwwwww#0U{������{[<0
��nnxz���������~znnnnnBFO[hqswta[SB:<7<<?B��������������������������������������������#39;891$
����TYagmz~�������aXRQPT��������������������xz|�����������zzxxxx����������������������������������������#*6CCDC@:6-*)Oh|����zn[OB6)��)6;B96*) ����������������"�������������������������������^ajmz��������zylda`^��������������������R[ht{xtqh[[TPPRRRR��������������������#/3<=<1/%#	)+15643.)	����������������������������������������������-3+��������##/596/#!##########��������������������������������������������������������������������������������������������������������������������������������������������������	
�������{������ynaXUMMQUZan{()25BEBBA5)(((((((((����������������������������������������]ahnz��������zna`Z]]Q[htpt���������o^KHQt����������tokkijklt��� ������������������
 !
�����������������������������������������������ghhtwuthd`ggggggggggU[[dgmt������tttg[UU��������������������NT[agt�������tg[UONN./7<HKONHH<4/.......OUan�������znha[UQOO#/0<;60#U[egt}�����xtg[ROPUUvz������zyvvvvvvvvvv��������������������!#0<>CHC<20,(#!!!!!!����������������������������������������E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E��#������������#�)�/�<�?�I�N�U�N�<�/�#�����������Ľнݽ�����������ݽнĽ����4�2�(�"����(�4�A�M�Z�[�[�Z�S�M�A�4�4��غֺպֺ���� �������������⾌����{�������������������������������f�c�]�e�f�s�w�����}�s�f�f�f�f�f�f�f�f�u�t�r�s�t�~�U�Q�P�O�U�`�a�e�c�a�U�U�U�U�U�U�U�U�U�U��ŽŽ���������������"����������Ƽr�l�f�d�a�f�j�r���������������r�r�r�r����������������*�+�/�*����������L�F�@�9�=�@�I�L�Y�[�b�d�Y�Y�L�L�L�L�L�L�(�������������5�A�B�G�H�F�>�5�(�F�:�1�4�:�B�F�S�l�x���������|�x�l�_�S�F�����	��	���"�%�/�0�/�"������������������	��"�/�;�>�<�;�"��� ������������������¿¹´¿������������
������������������������������������������ìêàÓÏÓàìóùû��ÿùìììììì�g�c�e�g�s�����������������s�g�g�g�g�g�g���m�Z�P�;���5�N���������������������������������������������������������������	�������	��"�;�m�~���������m�T�;�.����������������� ���������/�$�"�"������/�;�H�T�`�]�Y�T�H�;�/�g�Z�V�U�P�U�g�����������������������s�g�\�N�<�5�0�3�A�N�Z�g�����������������s�\ƎƁ�u�j�h�h�h�jƁƎƧƯ����������ƱƧƎ�M�E�A�4�0�0�4�?�A�E�M�Z�Z�_�^�Z�M�M�M�M�)�����)�6�A�B�C�D�C�B�6�)�)�)�)�)�)���y�m�`�\�U�Q�T�m���������ƿͿʿ��������	�����������	���"�'�.�1�8�.�"��	�	�v�s�{�t�N�J�Z�s����������������������v�T�P�N�N�H�A�?�H�T�a�f�m�m�o�o�n�m�i�a�T�[�[�W�[�g�h�tĀ�v�t�h�^�[�[�[�[�[�[�[�[�����������ÿĿʿʿĿ��������������������U�a�n�q�x�z�|�z�n�k�a�U�H�G�A�G�H�T�U�U��������������������������������������������������$�-�0�4�3�0�.�$�����ݽֽֽսݽ����	���������ݽݽݽݿ.�"����������;�T�`�m�z�{�y�l�`�T�G�.�6�3�4�6�@�B�E�O�P�O�N�B�6�6�6�6�6�6�6�6�Z�N�L�M�X�Z�f�s�����������������s�f�Z���������������������������������������������'�*�4�@�M�U�M�G�@�;�4�'�����f�Z�c���������!�.�9�6�!�����ּ����f�<�<�1�<�H�U�]�Y�U�H�<�<�<�<�<�<�<�<�<�<�@�?�6�4�1�4�@�G�M�Y�[�Y�X�M�@�@�@�@�@�@���������������������žʾооʾ���������ìæàÜÚØàáìîùïìììììììì�O�L�G�O�\�h�j�h�b�\�O�O�O�O�O�O�O�O�O�O���{�x�l�d�_�`�l�x�����������������������b�V�U�O�U�\�b�n�u�{ńŀ�{�{�p�n�b�b�b�b�~�k�j�q�~�����������ɺҺֺ޺غº������~�T�G�;�+�&�&�.�;�G�`�q�y�}�������v�m�`�T�������ĿϿѿӿݿ��ݿؿѿĿ����������������������������������������������������h�`�N�G�O�[�h�tāčĚĦĲĭĦĥčā�t�hƧƞƧƨƳ��������ƳƧƧƧƧƧƧƧƧƧƧ�;�8�/�%�#�%�.�/�;�H�T�V�U�U�V�V�T�H�;�;�0�'�#�<�I�Y�I�<�P�{ŠŹ������ŹŠ�{�b�0ĹĺĿ�����������
�����
����������ĹD�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�E*E&EEEEE$E*E7ECEPETEZE\E\EPECEBE7E*�ɺº������Ⱥɺʺֺ�����ֺɺɺɺɺɺ��������������������������	����'�'�(�'�����������������¿¹²§¦²¿�������������������ؼM�H�@�4�1�2�4�=�@�M�Y�b�f�r�{�{�r�f�Y�M�/�#������#�/�<�H�L�S�U�[�[�U�H�<�/FFFFFFFF$F1F1F9F1F-F$FFFFFF�������������¹Ϲܹ��������ܹйù����û����������ûлܻ��ܻܻлûûûûû�����������������������%�$����������������������������������������������������������������
���
�������������������������� ���������������������������ûлܻػлû��������������������!��!�$�.�0�9�:�=�D�F�:�6�.�!�!�!�!�!�! @ 2 - ;    & k � 0 7 D : D 4 i Y i B ` P Q Z g M 6 @ P : , D 8 H b u ` U * U # % G G P 5 V p < V G h R 3 X B  N R > L D R J V $ 9 d > L 2  ' 8  ' ` D Y $ h  �        [  �  X  �  �  0  �  U  �  @  {  �  �  �  �  p  �  V  �  .  i  �  �  �  2  �  �  �    �  �  _  �  �  �  �  #  �  T  �  B  �  �  H  v  �  �    �  �  U  ]  �  \    N  v  v  �  A  Q  l  [      F  �  �  �  �  O  4  �  �  t  �  CY  CY  CY  CY  CY  CY  CY  CY  CY  CY  CY  CY  CY  CY  CY  CY  CY  CY  CY  CY  CY  CY  CY  CY  CY  CY  CY  CY  CY  CY  CY  CY  CY  CY  CY  CY  CY  CY  CY  CY  CY  CY  CY  CY  CY  CY  CY  CY  CY  CY  CY  CY  CY  CY  CY  CY  CY  CY  CY  CY  CY  CY  CY  CY  CY  CY  CY  CY  CY  CY  CY  CY  CY  CY  CY  CY  CY  CY  CY  CY  �  �  �  �  �  �  �  {  a  A     �  �  �  9  �  �  m  (  �  �          	  �  �  �  �  Q    �  �  ?  �  �  p  �  �  H  X  m  ~  �  �  {  k  U  <    �  �  �    S  3  �  �  o  �  �  �  �  �  �  �  �  �  �  �  {  t  n  h  b  \  V  P  K  �  �  �  �  �  �  �  �  �  �  v  R  (  �  �  �  n  =  
  �  �  �  �  �  �  �          �  �  �  k    �  y    �  Z  \  ]  ^  `  a  a  ]  W  O  R  g  j  c  J  *  �  �  {     �  s  �  �  �  �  �  �  �  �  k  S  <  $  
  �  �  �  �  i  C  i  o  r  k  `  P  ?  .    	  �  �  �    
            �  �  �  h  J  +    �  �  �  K    �  �  8  �  8  �    p  �  �  �    1  C  L  T  W  T  B    �  �  m  +  �  �  1  �  �  �  �  �  �  �  �  �  �  �  �  o  T  9    �  �  �  �  v  N  C  9  '    �  �  �  �    `  A  #    �  �  �  �  �  `  X  Z  Y  S  G  9  )      �  �  �  �  �  �  �  �  k  Y  K  �  !  ?  ]  n  y  ~  z  o  b  Y  O  ?  %  �  �  ]  �  �  �  @  ;  6  1  ,  '  "             �   �   �   �   �   �   �   �  Z  P  G  F  K  F  8  )      �  �  �  �  �  d  ?    �  �  8  )           �  �  �  �  �  �  �  �  �  \  0      
    �  �  �  �  �  �  �  �  �  �  `  /  �  �  �  h  9    8  �  �  �  �  �  �    v  l  b  I  !  �  �  �  �  �  �  �  }  �  �  �  �  u  c  P  >  0  "      �  �  �  �  �  �  �  s  �  �  ^  "  �  �  r    �  |  >    �  �  b    �  H  �   �  �  �  �        �  �  �  �  �    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  c  5     �  �  k  =    �  �  �  �  �  �  �  �  u  h  W  ?  (    �  �  �  �  r  I      �  �  �  �  �  �  �  �  |  f  N  5    �  �  �  R    �  �  \  �                �  �  �  �  e  *  �  �  H  �  �  (  _  �  �  $  >  Q  p  �  �  �  �  �  }  R    �  e    �  	  �  �  �  �  x  m  c  X  N  C  7  '    �  �  �  �  {  \  =  �  �  �  �  �  �  z  k  [  I  7  %    �  �  �  �  �  �  �  ;  5  0  *  !    �  �  �  �  �  �  `  ;    �  �  e     �  �  �  �  �  �  �  �  �  �  �  �  q  U  5    �  �  �  V      	          $  3  *  +  E  .    �  �  �  �  F  �  �  o  y  q  Y  9    �  �  �  t  I    �  �  o  z  ?  �  N  �  P  B  3  "    �  �  �  �  �  b  A    �  �  �  h  ;     �  �  �  �  �    h  `  d  g  O  3    �  �  �  �  r  O  *    �  �  �  �  �  �  �  ~  m  \  J  9  (    �  �  �  �  �  q  g  L  )    �  �  �  Y  .  �  �  �  V  '  �  �  �  n  �  :  �  �  �  �  �  �  �  �  �  y  Y  4    �  �  f     �  j    �    )  2  /  %    �  �  �  o  5  �  �  Z  �  �  �  1  Q  �  �  �  �  �  �  �  v  h  W  E  1    	  �  �  �  �  w  T  �  �  �  �      %  &  &      �  �  s    �  �  $  f    �  �  w  e  O  9    �  �  �  �  s  O  *  	  �  �  �  �  �  $      ,      �  �  �  {  X  5    �  �  �  �  �  �  �  H  F  C  @  =  7  0  *  $               �  �  �  �  �  �  �  �  �  �  �  �  �  �  r  b  R  4     �   �   �   x   U   1  �  t  �  �  v  b  G  #    �  �  C  �  s  �  w  �  |  �   �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  m  J  .      `  �  z  `  A    �  �  �  c  2    �  �  q  >  	  �  �  �  �  �  �  t  h  Y  I  9  %    �  �  �  �  �  �  y  _  D  �  �  �  �  s  ^  G  -    �  �  �  �  f  :    �  �  �  P    �  �  �  �  �  �  �  �  �  �  �  �  �  �  s  ^  H  2        !           �  �  �  o  %  �  m    �  !  �  O    �  �  �  �  �  �  �  t  c  [  V  L  '    �  �  �  �  �  �  �      �  �  �  �  �  �  u  `  N  =  +    �  �  Y  �  [  �  �  �  �  �  �  �  �  |  b  B    �  �  �  I  �  U  �    �  �  �  �  �  �  {  j  Q  6    �  �  l    �  >  ;  �    �  �  �  �  i  G  &    $    �  �  �  �  [  3  	  �    5  �  �  �  �  �  �  �  �  �  �  �  �  v  H    �  v    �  G      �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  _  Z  Y  [  `  e  g  c  Z  G  2    �  �  �  �  �  �  �  m  �  �  q  a  C     4  Z  i  q  O  -    �  q    �  _  �  �  �  �  �  �  �  �  �  �  �  �  �  n  O  !  �  �  #  �  �  A  J  :  *      �  �  �  �  �  �  �  f  A    �  �  �  �  \  '  P  �       	  �  �  w     �  M  �    #    
�  	  0    �  �  �  �  �  ~  j  O  4    �  �  �  u  I    �  �  �  m  �  �  �  �  �  �  x  o  f  ]  U  L  C  ;  2  *  "      	  2  ?  K  X  B  &  
  �  �  �  g  B    �  �  �  �  [  3    �  �  �  �  }  s  h  ]  Q  G  <  2  *  "    �  �  �  �  �  b  �  �  	9  	c  	�  	�  	�  	�  	�  	�  	s  	,  �  M  �  /  �  ,  T  �  �  '  J  V  _  _  U  D  &  �  �  �  >  �  V  �  )  �  �  �    v  m  e  _  W  N  3    �  �  k    �  w  $  �  z  $  ?  L  Q  O  J  B  9  (    �  �  �  |  ]  *  �  �  K  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  z  Z  .    �  �  �  7  e  j  e  ^  S  D  0    �  �  �  �  q  A    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  [  M  @  2  $      �  �  �  �  �  �  a  :    �  �  �  c        
    �  �  �  �  �  �  �  �  l  W  C  (     �   �  �  �  �  �  �  ~  r  e  Y  M  C  ;  4  *        �  �  �  �  �  �  �  �  �  �  �    p  a  /  �  �  g  E     �  �  1