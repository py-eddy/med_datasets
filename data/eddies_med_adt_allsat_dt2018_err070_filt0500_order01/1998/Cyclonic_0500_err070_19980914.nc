CDF       
      obs    D   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?pbM���   max       ?�ȴ9Xb       �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M��   max       P�7       �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ��;d   max       =H�9       �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�ffffg   max       @F7
=p��     
�   �   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?��\(�   max       @v���Q�     
�  +|   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @.         max       @O@           �  6   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�        max       @��           6�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �n�   max       =,1       7�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�k�   max       B0�       8�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A��   max       B0?.       9�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       ?�wY   max       C��:       :�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       ?�ȿ   max       C���       ;�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          Y       =   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          7       >   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          7       ?$   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M��   max       PHe1       @4   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��ᰉ�(   max       ?��	�       AD   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ��G�   max       =H�9       BT   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�ffffg   max       @F1��R     
�  Cd   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?��\(�   max       @v���Q�     
�  N   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @&         max       @O@           �  X�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�        max       @�Ҁ           Y,   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         E|   max         E|       Z<   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�Fs����   max       ?�C,�zy     @  [L         	      	      P      
                     !            
            
      ,         E      	               
   ;      +   	      !   $   >   
         #   	   Y   	   1      	         G            7   -         *         N*JNbE5N+d�N>xN`y5M��P�7OO�\O[��N��lNC�Oh9N�X�N�3�O4��OQQ_O(BN+	�Ou�N�hQO��O��Ns�N�/OS��P&7�O+��O��O�V�NinN@+oO��N�(rP��N��?O@��P��N��O�VN�AWO�s�O�7�PA2�O�X�OO8�N���Oq��O���N���PK1�N��nO�*�O�t�N��UN��OSy:O��Op�zNf��N��YO�PU�N�d�OAF�O��WNA�aN���O�=H�9<�t�<u<o<o;D���o��o�ě��o�t��t��#�
�D���e`B�e`B�u�u��o��C���C����㼣�
��1��1��9X��j�ě����ͼ��ͼ�����`B�+�+�+�+�C��\)�t�������w�#�
�0 Ž49X�49X�49X�@��@��aG��m�h�m�h�q���y�#�����O߽�O߽�O߽�O߽�O߽�hs��hs����������-��������;d
&)*)









56BOOO[_[UOB64555555����������������������������������������3<IUbbcbUKIG<73333338<HUXUPH<:8888888888�������� ���������
/HU`URHD<3/#
"&/;BHHDC;7/"))6ABCNOQOFB6)!"))))BBGOU[^[OONB<>BBBBBB`hkjknt���������{th`�������������������������������������������
#/,/4/(#
����������������������������������������),)( "*6CMV[ZPC6*� mmz��}{zvmkijkmmmmmm�����������������������������������~}�~�|�����������||||||||��������������������O[ht|���}thd[YQOJIO��������������#./<@LHFGE</#������������������������
#/<BC?<0#���6BO[\h[OB76666666666��������������������������������������������


�����������<MUan����������zaU9<������

 �����#0<IUYXRPKI@<70#��������8#��������bgt�����������~tg`bb��������������������^agmrrqomka\XZ^^^^^^z}�����������zrryy{z���������������������������������������
./40+#
 �����S[gqt�����������n[SS���������������������������������������������"/5<5����MN[gt����tg\[NMMMMMM58?N[��������tg[B955!)-056765))���������������
���������������
���������������������������������~�����gjkt�������������tig������
����������5=FIUbekmnx{xaUIE<55mnor{�����|{wnmmmmmm��������������������)+3BN[[bjnmgb[NB5,')]ffmt{���������t[SS]#0:<@?<<720##!#07>IMQRPI<50#+/<Han���zncUH</+&'+!#%//57<><:/&#!!!!AHNUYaabaZUPHGC@AAAA�������������ۺֺѺκֺ������ֺֺֺֺֺֺֺֺֺ��/�/�&�#�#�#�,�/�3�<�C�F�@�<�/�/�/�/�/�/�H�B�<�5�<�H�U�Y�[�U�H�H�H�H�H�H�H�H�H�H�/�$�/�/�2�<�E�H�I�H�<�5�/�/�/�/�/�/�/�/�������������ʼ˼Ѽּ׼ּѼʼ����������������������������������������������������s�e�\�[�^�i�s������������� ����������s����&�'�&�)�1�B�U�[�e�[�V�R�O�B�6�)���������������������������������	����G�G�>�;�;�;�G�L�T�V�`�i�j�e�`�T�G�G�G�G�����������������ĿĿĿ��������������������������y�m�d�d�f�m�o�y������������������¿¿µº¿���������������������������˺ֺͺɺ������ɺպֺغ�������ֺֺֺ�ìáàÓÓÖÔàæìïùü������������ìÓÍËËÎÑÓÖàìðö������üùìàÓ�$������$�%�0�8�=�?�C�C�D�@�=�;�0�$�=�<�=�=�F�I�V�V�V�V�Q�I�=�=�=�=�=�=�=�=�;�4�"�	� �������	��"�.�6�<�?�B�I�N�G�;ÇÇ�{ÇÓàìõù��ùìàÓÇÇÇÇÇÇ�� �������!�-�:�F�B�:�7�-�'�!������������������������������������żY�T�N�T�Y�f�r�x��r�f�c�Y�Y�Y�Y�Y�Y�Y�Y�O�B�:�6�2�)��)�6�B�B�E�O�O�O�O�O�O�O�O�"�����"�/�;�T�a�l�d�a�V�T�H�B�;�/�"�6����������)�B�P�`�k�q�l�l�a�[�B�6�<�7�/�+�#� ��#�/�7�<�H�U�`�c�f�V�U�H�<���x�m�l�`�c�k�l�x�������������������������� ��������$�0�=�A�I�L�I�=�0�"������������ûʻϻʻû����������������������������Ƽʼּ׼ټּּʼ������������������������������������ʾϾվվʾľ��������������������������������������������������g�R�8�=�R�^�������������������������������������������������������������������������������������������������������������������������ĿͿͿ��������ѿ����z�x�t�q�m�l�m�o�z�������������������z�z�
��������0�<�I�b�n�{łńŀ�y�n�b�I�<�
��ƹƳƩƳ�������������������������������B�6�)�-�6�C�O�h�uƃƉƎƚƩƧƚƁ�h�O�B�T�H�E�M�P�T�j�m�s�r�x�z�}�}�����|�m�a�T�M�5�1�N�g�s��������� � �����������s�g�MEEE EEEE*E0E7ECEPEUEWETEPENECE7E*E���������������������)�.�1�)������������������
����$�&��������y�v�r�w�����������ʿѿ׿׿Ŀ������������׾˾��¾ʾ׾�������,�4�7�.�������������	����	����������������h�[�O�H�G�F�O�[�hāīĳ��������īĚā�h����$�(�4�A�E�M�U�Z�f�f�j�f�Z�M�A�4��f�`�k�r�}���������������㼱���r�f�f�~�������|�r�n�o�f�L�@�>�?�=�?�M�W�d�fŹŭŹŽž����������������������ŹŹŹŹ�����������������������������������b�U�I�;�?�9�.�#��#�>�I�W�b�k�o�t�s�n�bĿĳİİķ�������������
����
������Ŀ���������}�v�x�����������ûлջһлû����l�f�_�S�G�K�S�_�k�l�x��x�n�l�l�l�l�l�l������������$�'�0�4�;�4�*�'��������U�H�<�*���#�/�<�H�a�zÇÐÛÝØÇ�n�U��²�p�g�\�Y�b�t¦¼�����������ؽ������������������ĽнӽннĽ�����������߽�������(�4�?�<�4�/�#������龱���������������������ʾԾ۾߾ܾ׾ʾ���E�E�E�E�F FFF#F$F&F$F FFFE�E�E�E�E�F1F*F1F5F=FIFJFVFWFcFfFcFYFVFJF=F1F1F1F1�.�)�$�-�:�G�`�l�y�����~�y�l�h�`�S�G�:�. E M F w q T  ] ^ 5 6 @ ' V X ] ] ] k { B M < c 4 @ ) # 9 c E  7 b z ` K g ^ o L v [ D W E 2 E t  b p } ] t m ) ( : h 9 m @ P x w S h  Z  �  A  �  �  >  g  (  �  �  d  e    �  �  �  �  `  U  �  ^  v  �  Z  �  �  i  &  H  �  g  '  �  L  F  �  �  3  K  �  &  �  �  ,  �  �  �  �  *  Z  C  V  �  �  A  9  �  �  v  �  �  Z  
  �  �  �  �  �=,1<e`B;�`B;o%   ��o��ě���o�e`B�e`B��9X��9X��t������<j��󶼣�
��P��/��h�8Q�+����㽇+�L�ͽT����vɽ+�C����#�
�e`B�'0 Ž�vɽ,1���w�<j�m�h��hs������
=�Y��P�`�m�h��aG��n���+��"ѽ�t���O߽�������\)���
������P�   ��h�\�\�o��l���S��oB�UBu�B!B��B&��B+GB�BT&A�k�B�<B��B��B��B ��B�nB�_B/�B�B0�A���B,fBm�BD�B�FB��B��B�7B L�B��B�gB 4�B!�BD�B��B#}�B&�B�'B
K�B��A�5�B ��B��Bb�B�1B	��B)��B+�B��B	U,B	�B�*B-/�B"QCB*�B
�B
�%B�B'CEB(��B)�aB�B
LJB%��B%��BD�BB�B�6B
B�YBDB! 3B3]B&��B?�BAB��A��B=�BNB�hBҭB ǆB�B�\B;�B+�B0?.A��uB,=�BH�B?kB�B�B?4B�B @�BM2B��B 8�B!zFBF�B��B#�;B&?�BB�B
��B��A���B?tBG�BA�B�(B
TB* XB+#�B��B	�B	T�B5�B-?�B"�jB[�B
�3B
_{B?�B&�BB(��B)��B��B
��B%@�B&D�B�6B��B��B��@A��A���AĤ�A�	�@�A�ſA�5\A� �A���Ag�Au܊Am�}A�0�@<O�A�A��9B
-�B8zA^�FA�g�@k3?A��`@�3�A��A�fgA��AĎ�@�B	�?@�-4@�H�AM8�A��A��PA���A��Ay*�A��A��BO�BA�A��1A��C��zA��A�̂As�
AY
�AY�.A���A:��@���?�wYA�\�B��A�*sA�H6@�� @��@�I�AƟ@A���A#hA2�.AN��C��uC��:A�]@C��A��kAĎ)A��@��A���A�^�A�/A�5AePAv��Al�A�}�@?U-A�}�A�?�B
B�B?�A]K�AʁQ@lFjA�t�@�A֘�A���A��Aă�@�bIB	�y@��T@���AM�|A�Z�A�@�A���A�ZAx��A���A�z�B�&B��A���A���C���A�xA���Ar!CA[QAZ(�A� A<�A-:?�ȿA�u1B>�A�~�A�&@���@�M�@�A.A�cYA�S�A"��A5R�AN��C��C���A��         	      	      Q                           !                              -         E   	   	            	      <      +   	      !   $   ?   
         $   	   Y   	   2      	   	      H            7   .         *                              1                                                         )                        /         )      !      '      7               #      -      -   "            !            !   /                                       )                                                         )                        /         #      !      '      7                     #      -   "                           +                  N*JNbE5N+d�N>xN.�)M��PHe1O-�O[��N��lNC�N�N��PN�3�OiO��O��N+	�OA��N�hQN�G�N���Ns�N�/OS��P&7�N]�Oj:�O;DUNinN@+oN�l-N�(rP��N��?O@��OΘ�N��O�VN�AWO�s�OZ+PA2�O� O}tN���Oq��O@rCN���O�d�N��nO�*�O�t�NqiN��OSy:O�$-ONz�Nf��N��YO��P��N�d�OAF�O9Q�NA�aN���O�W  M  �  N  �  �    �  ,  �  �  o  �  �    w  P  L  :  5  �    �  X  0  0  �  �    �  �  �  �  �  �  |  }  �    5  D    {  (  
�  �  �  J  "  �  
�  �  �  �  �  �  P  
  �    {  �  X    �  �  �    �=H�9<�t�<u<o;�`B;D�����
�t��ě��o�t��e`B�49X�D����o��9X��o�u���㼋C����
�����
��1��1��9X�C���`B�'��ͼ�����h�+�+�+�+�49X�\)�t������,1�#�
��o�<j�49X�49X�u�@����w�m�h�m�h�q���}󶽅���O߽��罏\)��O߽�O߽��罕�����������j��������G�
&)*)









56BOOO[_[UOB64555555����������������������������������������:<IU]aUI<8::::::::::8<HUXUPH<:8888888888��������������������#/<@HMOH?<80/%""&/;BHHDC;7/"))6ABCNOQOFB6)!"))))BBGOU[^[OONB<>BBBBBBst����������troqssss�������������������������������������������
#)/%#
����������������������������������������������),)($*6CIOSWVOKC6*mmz��}{zvmkijkmmmmmm����������������������������������������|�����������||||||||��������������������O[ht|���}thd[YQOJIO��������������#/186/#��������������������
#&/5<=<7/#
 ��6BO[\h[OB76666666666��������������������������������������������


�����������<MUan����������zaU9<������

 �����#0<IUYXRPKI@<70#������������������bgt�����������~tg`bb��������������������^agmrrqomka\XZ^^^^^^z}�����������zrryy{z�������������������������������������������
#%#
������V[gtv����������}t[VV���������������������������������������������������MN[gt����tg\[NMMMMMM=BN[���������tgND?<=!)-056765))���������������
���������������
���������������������������������~�����gjkt�������������tig�����������������7>BGIUbgknpvrb^UF=67mnor{�����|{wnmmmmmm��������������������25BN[_egfda[NB51-+.2^ggnt~���������t[UU^#0:<@?<<720##!#07>IMQRPI<50#-/6<HUagmia^UH</.()-!#%//57<><:/&#!!!!AHNUYaabaZUPHGC@AAAA�������������ֺܺѺκֺ������ֺֺֺֺֺֺֺֺֺ��/�/�&�#�#�#�,�/�3�<�C�F�@�<�/�/�/�/�/�/�H�B�<�5�<�H�U�Y�[�U�H�H�H�H�H�H�H�H�H�H�/�$�/�/�2�<�E�H�I�H�<�5�/�/�/�/�/�/�/�/�����������ʼϼּϼʼ����������������������������������������������������������������s�g�e�h�s���������������������������6�*�)�%�(�)�)�-�6�B�O�[�`�[�O�N�O�O�B�6��������������������������������	����G�G�>�;�;�;�G�L�T�V�`�i�j�e�`�T�G�G�G�G�����������������ĿĿĿ������������������m�k�j�l�m�y�y�}���������������y�m�m�m�m����¿·»¿���������������������������˺ֺͺɺ������ɺպֺغ�������ֺֺֺ�ìåâààØÙàìùûý����������ùììàØÓÐÏÏÓÛàçìñùúÿýùôìà�$�������$�'�0�=�B�B�C�@�=�:�0�$�$�=�<�=�=�F�I�V�V�V�V�Q�I�=�=�=�=�=�=�=�=�"��	���������	��"�*�.�3�9�<�?�C�;�"ÇÇ�{ÇÓàìõù��ùìàÓÇÇÇÇÇÇ�����	���!�-�:�A�=�:�3�-�"�!����������������������������������������Y�T�N�T�Y�f�r�x��r�f�c�Y�Y�Y�Y�Y�Y�Y�Y�O�B�:�6�2�)��)�6�B�B�E�O�O�O�O�O�O�O�O�"�����"�/�;�T�a�l�d�a�V�T�H�B�;�/�"�6����������)�B�P�`�k�q�l�l�a�[�B�6�H�G�<�4�:�<�H�U�X�\�U�I�H�H�H�H�H�H�H�H�x�r�o�l�g�f�l�x�����������������������x����	����$�0�4�=�C�G�C�=�9�0�$�������������ûʻϻʻû����������������������������Ƽʼּ׼ټּּʼ��������������������������������������ʾ˾ѾҾʾ��������������������������������������������������g�R�8�=�R�^�������������������������������������������������������������������������������������������������������������������������Ŀѿ���������ݿѿ����z�x�t�q�m�l�m�o�z�������������������z�z�
��������0�<�I�b�n�{łńŀ�y�n�b�I�<�
��ƹƳƩƳ�������������������������������B�6�)�-�6�C�O�h�uƃƉƎƚƩƧƚƁ�h�O�B�T�I�H�G�K�O�R�T�a�m�y�z�|�|���z�z�m�a�T�M�5�1�N�g�s��������� � �����������s�g�MEEEEEE#E*E7ECEHEPEQEPENEFECE7E3E*E�� ���������������������������������������
����$�&��������y�v�r�w�����������ʿѿ׿׿Ŀ������������۾׾;Ծ׾���	���!�*�'�"��	��������������	����	����������������h�[�P�M�L�O�V�[�h�tĒěĨıĲĦĚč�t�h����$�(�4�A�E�M�U�Z�f�f�j�f�Z�M�A�4��f�`�k�r�}���������������㼱���r�f�f�~�������|�r�n�o�f�L�@�>�?�=�?�M�W�d�f���������������������������������������������������������������������������b�U�I�;�?�9�.�#��#�>�I�W�b�k�o�t�s�n�b����ĿĽĹĹ����������������
�	�������ػ������������{�����������û̻лԻлû����l�f�_�S�G�K�S�_�k�l�x��x�n�l�l�l�l�l�l������������$�'�0�4�;�4�*�'��������H�C�5�0�6�<�H�U�a�n�zÈÒÔÏÇ�z�n�U�H��²�s�g�^�[�d�t¦º������
�����ؽ������������������ĽнӽннĽ�����������߽�������(�4�?�<�4�/�#������龱�������������������ƾʾѾؾܾؾ׾ʾ���E�E�E�E�F FFF#F$F&F$F FFFE�E�E�E�E�F1F*F1F5F=FIFJFVFWFcFfFcFYFVFJF=F1F1F1F1�.�*�%�.�:�G�S�`�l�y���|�y�l�g�`�S�G�:�. E M F w e T  Q ^ 5 6 = & V ` U U ] \ { G N < c 4 @ ' % / c E   7 b z ` ( g ^ o L W [ @ i E 2 T t ( b p } , t m  ) : h 2 l @ P ; w S f  Z  �  A  �  z  >  A  +  �  �  d  �  �  �  P  g  w  `  �  �  
  7  �  Z  �  �  k  �  �  �  g    �  L  F  �  �  3  K  �  &  �  �  4  ]  �  �  �  *  :  C  V  �  �  A  9    �  v  �    2  
  �  �  �  �  x  E|  E|  E|  E|  E|  E|  E|  E|  E|  E|  E|  E|  E|  E|  E|  E|  E|  E|  E|  E|  E|  E|  E|  E|  E|  E|  E|  E|  E|  E|  E|  E|  E|  E|  E|  E|  E|  E|  E|  E|  E|  E|  E|  E|  E|  E|  E|  E|  E|  E|  E|  E|  E|  E|  E|  E|  E|  E|  E|  E|  E|  E|  E|  E|  E|  E|  E|  E|  M  ;  (       �  �  �  �  �  m  O  1    �  �  �  �  �  w  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  N  O  Q  Q  N  J  ?  4  (         �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  }  W  1    �  �  o  �  �  �  �          �  �  �  �  �  �  i  L  .    �  �       �  �  �  �  �  �  �  �  �  y  a  J  2    �  �  �  �  �  K  �  �  �  �  �  �  O    �  }  0  �  {    �  �       �  �        )      �  �  �  }  M  %    �  �  Y    K  �  �  �  �  �  �  s  X  ?  %        �  �  �  �  l  @    �  �  �  �  y  q  i  _  S  H  ;  -             �   �   �  o  h  `  Y  P  F  <  2  %      �  �  �  �  j  C     �   �  z  �  �  �  �  �  �  �  �  �  �  �  n  J  *    �  �  �  @  �  �  �  �  k  O  ;  "    �  �  �  m  /  �  �  r  ?  C  o    �  �  �  �  �  �  �  �  �  m  V  @  (    �  �  �  �  �  >  `  u  w  o  e  Z  J  6      �  �  �  �  �  �  �  �  �  �     1  E  O  O  D  0    �  �  `  
  �  Y  �  �  G  �  K  2  H  F  =  1  #    �  �  �  �  G  �  �  B  �  v    �   �  :  (      �  �  �  �  �  w  ]  D  '    �  �  �  i  6    4  0  ,  0      �  �  �  �  |  U  8    �  �  Q  �  �  �  �  �  �  �  �  m  W  >  $  	  �  �  1  �  �  A  �  �  a                       �  �  �  �    S  "  �  �  �  c  v  �  �  �  z  {  �  �  �  �  �  a  7    �  �  v  Q  �  K  X  O  F  <  1  %      #  ,  !    �  �  �  `  5  	  �  �  0  )  %  2  F  q  �  �    /  [  �  �  �  �    ?  a  �  �  0  "    �  �  �  �  �  �  w  _  G  /    �  �  �  �  Y  �  �  �  �  j  *  �  �  m  #  :      �  �    �  T  �  �   |  �  �  �  $  o  �  �  �  �  �  �  �  ~  2  �  ^  �    U  �  �            �  �  �  �  \  2    �  �  �  m  ?    `  
�  '  f  �  �  �  �  �  `  5  
�  
�  
=  	�  �  <  l  �  �  %  �  �  �  z  l  _  P  B  3  #      �  �  �  �  X  0    �  �  �  �  �  �  �  �  �  �  d  D  !  �  �  �  s  ?  
  �  �  t  z  �  �  ~  x  q  h  ^  R  F  8  +         �  �  �  �  �  �  �  �  �  �  �  �  �  w  m  a  M  8  #    �  �  �  �  �  �  t  Q  /      �    	  �  �  �  �  {  Z  1  �  �   �  |  j  X  D  /         �  �  �  �  �  L    �  �  F   �   �  }  r  g  \  O  B  2      �  �  �  �  �  �  �  �  �  z  k  �  c  �  �  �  �  �  k  G  %    �  �  �  5  �    [  �  F      �  �  �  �  �  �  �  �  �  �  �  �  v  `  K  0    �  5  ,          �  �  �    L  =     �  v    �  c  w  {  D  ,    �  �  �  �  �  �  q  ^  L  ;  1  (  �  �  �  c  9    �  �  �  �  �  �  s  O  *    �  �  c  *    �  �  {  F  �    {  v  e  F    �  �  p  ,  �  �    "  �    �  $  �  (      �      �  �  �  }  K  "  /  %    �  �  G    `  
.  
Q  
h  
�  
�  
�  
�  
�  
�  
�  
;  	�  	Z  �  M  �  �    j  �  �  �  �  �  �  �  �  �  �  �  �  �  t  [  D  )    �  �  i  �  }  y  s  j  `  U  I  <  .      �  �  �  �  �  �  �  n  J  @  2       �  �  �  �  �  y  \  =    �  �  �  o  8    �  �  �  �        "       �  �  i    �  (  �  q    E  �  �  �  �  }  l  [  J  =  0  '  !    	  �  �  �  �  �  {  
u  
�  
�  
�  
�  
�  
�  
�  
J  	�  	�  	Q  	   �  N  �  �  �  �  S  �  �  �  �  �  �  �  �  �  �  �  �  �  �  }  n  \  /  �  x  �  o  �  Q    �  |  ;  S    �  �  a    �  s  �  K  �   c  �  �  �  �  �  z  ^  S  >  #    �  �  �  �  �  a  <  �  V  i  y  �  �    v  p  j  _  T  H  9  *        �  �  �  �  �  �  �  �  z  ]  :    �  �  �  �  �  �    n  ^    �   �  P  /    �  �  �  �  �  �  �  �  �  �  q  T  2    �  �  p  	�  	�  	�  	�  
  	�  	�  	�  	�  	�  	v  	3  �  a  �  N  �  w  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  l  O  0    �                  �  �  �  �  �  �  �  �  �  �  �  �  {  x  t  p  k  b  Z  Q  I  @  6  -    �  �  �  �  p  V  =  (  f  |  �  �  �  �  {  n  V  3    �  L  �    W  �  �  �  9  W  R  F  9    �  �  �  �  j  3  �  �  )  �    l  w   �    �  �  �  �  �  �  n  P  3    �  �  �  �  �  W    �  �  �  �  �  f  I  *    �  �  �  _  <  1    �  �  �  K  
  �  A  �  �  �  �  �  �  g  4  �  �  T  �  �    �  (  �  �  �  �  �  �  �  �  �  o  U  ?  0      �  �  �  T  
  �    �    �  �  �  �  g  ?    �  �  �  N    �  �  m  (  �  �  N  �  �  �  �  �  �  X  '  �  �  �  T  "  �  �  o  2  J  �  \