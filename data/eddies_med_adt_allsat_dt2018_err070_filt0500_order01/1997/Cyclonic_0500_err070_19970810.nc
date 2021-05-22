CDF       
      obs    K   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?��-V     ,  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M�l�   max       P�ʻ     ,  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       �ě�   max       =o     ,      effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>���Q�   max       @F<(�\     �  !0   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?��Q�     max       @v}��R     �  ,�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @.         max       @R@           �  8�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�9        max       @���         ,  98   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �+   max       <�     ,  :d   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�[�   max       B0P0     ,  ;�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A��A   max       B0@\     ,  <�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >��   max       C��O     ,  =�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >��h   max       C��%     ,  ?   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          t     ,  @@   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          E     ,  Al   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          -     ,  B�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M�l�   max       PJ3     ,  C�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?���@��   max       ?Ҟ�u     ,  D�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       �ě�   max       =o     ,  F   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>���Q�   max       @F0��
=q     �  GH   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�(�X    max       @v}��R     �  S    speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @'         max       @Q            �  ^�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�9        max       @�)�         ,  _P   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         C�   max         C�     ,  `|   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��,<���   max       ?Ҝw�kP�     p  a�            *                           
   t            7                  '      $   
   8         	                        c   !      	   
               
            $      "   &         3         
                           '         
      N=.tO��NqO�g�O0��N��N�N�4�O
�O���NJ�|O���O�Pq�5O�?BOyK.P�iP!�OO`��O�@�N�ON�h�N^��O��KO���Os=Ǹ�P�ʻO�Or''N�IFN�NOB}N���N���O��O�āO�4�P^6�On>aN���ṄN��mN�5�N�Od]�OHE%N��O��O�_N��gOH�5O�~O��PO���O�NK%{P�N؉�Nȡ�O�UN2<�Ou�(OBKHOf��Ol|�O ��NN��M�l�O�7N'�N�~)N��N�r*N�ɼ=o<e`B;D��;o��o�o��o�ě��ě��o�t��T���e`B�u�u��o��C���t���t���t����㼛�㼛�㼛�㼬1��1���ͼ��ͼ�/��`B��`B��`B��`B��h��h���o�o�+�+��P������w�,1�,1�0 Ž0 Ž49X�<j�<j�@��Y��Y��]/�]/�ixսq���q���u�y�#�}󶽁%��+��+��+��C���hs������-���T��� Ž\�ě���������������������bgmt���������vpnjjeb����������������������������������������	!)5BNR[^a`[NB6	<<BIUbha\URJI<;<@<;<tz����������|zwqontt�������������������LO[[htv~{tsoh[ONKLLLju��������������|olj����������������������������������������������������nUI<42+*' $/Ucn��tn�������
������
���������
NSXg������������g[NN��������������������fot������������h^^_f$*6COU\_]YOC6*���		����������X[agt|����wtg][ZZXXXW[fggjokg[RTWWWWWWWW��(./---)�����";HTUlpywqaTA;/"��������������������TUanz������zqnaWUPTT�������+��������{�����������������z{<BO[hjlihggijih[OFC<st��������������vtrsst��������}tqossssss��������������������V[ghkorqg[ZUTTVVVVVV36BCJOPSQOKB>6630/33��������������������GNgt������tlgaa[[NDG8<MXn��������{nbI<48���),11$("������>HOX[htrvtnsha^TB:6>>BNO[gt�~tg[NB>>>>>>�����������������������������������

�������������������������������)6>B@6)���������

 ��������������  ��������������������������������������������������������������������������������������{{�����������������{�����
	���������N^t����������tsg^NLN����������������������������������������������	����������������������������GHKUU]adntwsnfaWUHGGyz|������������zywxy��������������������z~��������������zvuz
#/56:;<<</#
	#/<AHLPVVROHA</)#5:=:<HPUantvsnaH<6/5��������������������������������������������������������������
#./<?F>4/)#
�������������������������458BDNQUVRNB?6544444egkot{���������tsgeeaX\ainxz���zznaaaaaa�Z�Y�V�R�Z�g�s�t�t�s�g�g�Z�Z�Z�Z�Z�Z�Z�Z���������¿ƿѿݿ����������ݿѿĿ��H�D�<�;�<�H�O�U�[�X�U�S�H�H�H�H�H�H�H�H�#����������	��#�6�<�H�J�I�U�o�`�U�H�#�g�b�`�Z�W�S�U�X�Z�b�g�s�����������j�i�g�ʼɼ����������ʼּ����������ݼּʺɺź��������źɺֺ����������ֺɺɿݿڿѿѿͿϿѿݿ������������ݿݿݿݿm�k�`�`�Z�`�`�m�y�����������������y�m�m�����۽ֽ׽νʽս�����$�'�$������������������������������������Ŀݿ����5�D�>�5�(�����ѿĿ����~�y�o�m�e�a�m�y�����������������������N�n�x�u�f�4��ݽĽ������y�~�������A�N����������ùõù�����������	������������������׾ʾ������������ʾ׾���/�	�������������"�(� �!�*�*�0�L�I�<�/�/�'�#�%�#�'�.�;�T�m�������������m�T�H�/��׾ѾоԾ׾޾����	���"�-��	���������ݾؾݾ����	��'�:�<�7�-�"��	���	������������	�����	�	�	�	�	�	�	�	���������(�(�4�;�<�A�G�A�4�(���Z�P�M�M�M�Z�f�s�v�|�s�f�Z�Z�Z�Z�Z�Z�Z�Z����ҺȺɺֺ�����-�>�<�:�2�-�'�!���������������������	�������������������������!�-�:�F�_�l�u�w�l�_�S�-�������������������������������������������غ��~�p�~�����Ļ�:�F�_�x��l�:����ֺ������{�z�q�a�\�^�a�m�z�������������������������������������������������������������	����	���	���!�"�%�'�'�(�"���	¿¾¸»¿������������������¿¿¿¿¿¿�������������������������u�j�p�uƁƎƚƧƲƨƧƚƎƁ�u�u�u�u�u�u�S�P�R�S�_�i�l�x���������������x�l�_�S�S���׾ʾǾ¾о��	��"�.�;�=�;�1��	���������������������5�B�U�N�B�5�)������������s�b�Z�V�e�������������������������r�f�J�4������'�@������������������r�ܹù��������ùϹ���������������ܿ	������������	����	��	�	�	�	�	�	���������	���"�-�.�"��	������������������������������������������������������������ �����$�'�1�,�'�������ìçéìù������ÿùìììììììììì�T�J�;�6�3�2�1�5�;�G�T�a�i�n�r�o�m�m�a�T����� �������*�6�<�G�F�C�6�*���čĆĉĈčĚĦĪĳĹĶĳĦĚčččččč�����������������������!���������H�@�B�H�L�T�T�a�j�m�t�z�}�z�y�m�i�a�T�H�������������������������������������������������������'�1�8�;�<�3�'��@�4���
�4�M�Y�f����������u�k�_�Y�M�@�������������ʼ����	������ʼ�����ĳĦěėĕĕĚĳĿ������������������Ŀĳ����������� �(�5�6�<�7�7�.�'������������������������������������������6�O�tčĚĢĩħĝč�t�B�)�����ĿĳİĳĿ�������������������������̿ѿϿĿ��������������Ŀѿ׿ݿ�ݿܿӿѿѾ��������������������ʾ׾���Ҿʾ������=�4�=�I�I�U�V�b�e�h�b�V�O�I�=�=�=�=�=�=ƳƧƎƇƁ�|ƁƚƧƳ������������������Ƴ���ݿӿҿڿݿ��������%������E�E�E�E�E�E�E�E�E�FFF$F2F=FCF<F1F!E�E��Z�N�A�5�&����"�*�5�A�N�Y�a�i�u�s�g�Z�I�@�>�=�6�=�I�V�b�o�{�{�|�{�p�o�b�V�I�I������������$�%�$�!����������������L�G�L�L�Y�e�f�m�e�Y�L�L�L�L�L�L�L�L�L�LED�D�D�D�D�D�D�EEE*E-E7E=ECE@E0E*EE�� ���������
������������S�Q�S�S�_�`�l�y���������������y�w�l�`�Sùïìèìôù������������������ùùùù�/�,�#���
� �
���#�/�<�?�?�<�<�2�/�/��ܻлʻû��������ûлܻܻ������� ; B U C l \ ] 1 . c L d S c H * j 4 & $ n 7 Q F ` Y � s I l �  1 ` O K J E c B � 5 V ; J F 9 6 > L K O t M 7 � K m X , ` � g ( 4 B 0 r Q 4 P u A ( ^    h  �  B  \    �    �  *  �  f  _  R  �  $  �  �  �  �    H    h  a  Y  A  R  �  r  I  �  �  m  �  �  �  
  (  �    =  �  �    A  �  �  �  D  ?  �  �  �  �  i  �  h  4    �  p  �  d  �  �  �  *  �  !  ;  8    �    �<��T���D����w�D���#�
��o�D���T���ě���o��P�ě����t��#�
�'���\)�<j��9X�ě��ě��m�h��w�e`B�\)���T�+�H�9��P�C��\)�#�
�<j�D���P�`�m�h�+��+�'@��@��L�ͽ@���+�ixսT���ixս]/�T�����O߽�-��^5��7L�u��;d���P�����\)��7L���{��vɽ��-���-�����h��1��E��ě���"ѽ�/BѪB
ٝB!-vB�BB'oB�<B �kB�hB��B�8BQ�B*.B&ۇB�{B#�B
�&B��B�wB0P0A�[�B	uiB	�B[�A���B,E�B�WB":�B2uBB�eB��B,��B	�B\B�OB	�5B(�B�BF�B	&�B�B��B#�GB�%B�XB{�B��B�BB��BZBH1B*I}B-N�B	�)B*�BB"B5mB��B7�B�B�B n�B��B��B��Ba�B$OBf�B.5B�$B��B�!B
$BɯB	�B
ɹB!1�B4#B�%B&�QB�	B �bB�yBAB�~B>�B*@B'A B@dB#�~B	�^B:�B��B0@\A��AB	��B��B��A�x
B,��B@�B#A�B<�B�B ��B��B,��B�0B?�B��B	hB)>3B�^B?�B	3B�sB��B$�B��B¸B�B�$B��B��BzTBQ�B)�
B-��B	�.BZ�BF�B?�B��B?�B�'BBB :�BAGBEKBCBD�B��BD�B�+B��B�`B��B
A8B�pA��A~��A��A��;A�6�@�<@?	A~	�AmL�A/�A��TA�i{Ao��A-:A�-�AT��A��A���AX��AZz;A�|A68<A@��@W��A���@|�6A���@U��A�f�A�W�A���A��6A��pB�)@���AY��A�f�A�_@ᓽ>��AY�vA\J�A�ش@���A�q"A��A���A߭hA��TA�[�A��?�g�@�%@�"�A�5�A�W�B�pA�:?A�OAy*WAN5FBo\Bu�A��"C��OA�)1B��B�W?�+�C�qNA/��A3�A΢HA���@�Z�A�%�A21AąhA� WA�e@��h@;F>A.Am�8A0��A���A��4AmKA-;�A҃JAT��A��DA�~AV��AZ�A� �A6��AB�@K��A���@���A�{@Z�A�~�A��A�QA��%A��.B��@� #AYےA�`xA���@� 3>��hAYm2A\�	A�a~@��1ȀkA��8A���A�	�A�~�A���A���?q��@�A ��A�kA��B��A֧1A䃨Ay��AM��B/OB>�A�C��%A��B@B	?{?��C���A0��A�"AΆ�A���@���            *      	                        t            8                  (      $      9         
                        d   "      
   
               
      	      $      #   &         3         
                           (                           '                  !      '      :         -   %                     '         E                        !   %   %   7                                          %   !            +                                                                                             '         -                        %         -                              %   +                                             !                                                               N=.tO{SNqOgr~O�N��&N�?%N�4�O
�OS�5NJ�|OK�O�O��}O)�*OyK.P�iO���O`��O#N�ON��FN^��O�\�O�T~OW��Ǹ�O���O�OK��N�IFN�NOB}N���N�[{OPO3��O�4�PJ3N�2HN@N�k9N��mN�ȈN�O;O�OHE%N��O��O�_NJ΁O�)Ok�RO��PO��}O�NK%{O��N؉�N�_O�UN2<�Ou�(OBKHOf��Ol|�N��NN��M�l�ON�N'�NqhuN��N�\�N�ɼ  �  k  �  �     �  �     -  �  �  �  �  
=  �  �  �    C  '  �  �  <  f  a  �  _  �  �  �  �    �  �  A  �  _  �  ]    z  /  :  F    �  O    [  �  �    �      Y    a  K  a  \  �  �  �  �    �  �  j  	  4  I  }  b  \=o;D��;D���u�D���D�����
�ě��ě��49X�t��ě��e`B�u���㼃o��C��C���t��������㼣�
���㼼j��9X�ě����ͽ49X��/����`B��`B��`B��h���\)��w�o�Y��49X����w���,1�,1�8Q�0 Ž0 Ž49X�<j�@��Y��]/�Y��e`B�]/�ixս�O߽q���y�#�y�#�}󶽁%��+��+��+��O߽�hs�����1���T��1�� Žě��ě���������������������{��������������zwv{{����������������������������������������()5BLN[_^[RNMB95,) (<<=DIUbgb_ZUOIB<<<<<uz����������{zyspouu�������������������LO[[htv~{tsoh[ONKLLLqu����������������uq�����������������������  �������������������������������8<IUZenqrfbUI<965658�������

�������
���������
NSXg������������g[NN��������������������fot������������h^^_f"*6CIOSVTOMEC6* ���		����������Z[gt���~vtg][YZZZZZZW[fggjokg[RTWWWWWWWW���&--+**)%��";HTbjoxumaTB;/��������������������TUanz������zqnaWUPTT�������	��������{�����������������z{>BO[ghkhgeehg[TOHEA>st��������������vtrsst��������}tqossssss��������������������V[ghkorqg[ZUTTVVVVVV56=BIOOQOODBB9641055��������������������P[gt�������ythg_[UPP8<MXn��������{nbI<48��� &*)������NO[hllmnnh[ONGGINNNNR[gt{ztg[RRRRRRRRRRR��
	����������������������������������

��������������������������������)6:?<62)�������

 ��������������  ��������������������������������������������������������������������������������������������������������||������
	���������P[`t����������tg`[NP�����������������������������������������������������������������������������LUV^adnsvrneaZUILLLLyz|������������zywxy��������������������z~��������������zvuz
#/56:;<<</#
	#/<AHLPVVROHA</)#5:=:<HPUantvsnaH<6/5��������������������������������������������������������������
#/093/,%#
�������������������������������458BDNQUVRNB?6544444gglpt��������utgggggaX\ainxz���zznaaaaaa�Z�Y�V�R�Z�g�s�t�t�s�g�g�Z�Z�Z�Z�Z�Z�Z�Z�ѿ̿˿οѿ׿ݿ�������
������ݿѿ��H�D�<�;�<�H�O�U�[�X�U�S�H�H�H�H�H�H�H�H�
����������
��#�/�<�B�D�J�K�H�<�/�#�
�Z�Z�W�W�Z�[�g�s���������������{�s�m�g�Z�ּмʼ����������żʼּݼ����ټֺּּּɺƺ����������ɺֺ�����������ֺɺɿݿڿѿѿͿϿѿݿ������������ݿݿݿݿm�k�`�`�Z�`�`�m�y�����������������y�m�m������޽۽ݽ߽��������"� ����������������������������ݿѿϿοѿݿ�������������꿆�~�y�o�m�e�a�m�y���������������������������������Ľݽ��(�3�C�D�+����ݽ��������������������������������
��������������׾ʾ������������ʾ׾���/�	�������������"�(� �!�*�*�0�L�I�<�/�;�1�-�.�0�9�H�T�a�m�z�~��������n�a�H�;��׾ѾоԾ׾޾����	���"�-��	�����	�����������	���"�$�+�*�"���	�	������������	�����	�	�	�	�	�	�	�	�������!�(�4�8�:�4�/�(�������Z�P�M�M�M�Z�f�s�v�|�s�f�Z�Z�Z�Z�Z�Z�Z�Z����ֺʺ˺ֺ�����!�&�-�4�3�/�-�!����������������������	����������������� ����!�-�:�F�S�_�l�r�t�l�_�S�-�����������������������������������������غ����������Ǻۺ����!�:�X�N�:�!���κ����{�z�q�a�\�^�a�m�z�������������������������������������������������������������	����	���	���!�"�%�'�'�(�"���	¿¾¸»¿������������������¿¿¿¿¿¿�������������������������u�j�p�uƁƎƚƧƲƨƧƚƎƁ�u�u�u�u�u�u�S�R�S�T�_�l�l�x��������������x�l�_�S�S�	����׾ξ׾پ����	��"�.�1�,�"��	���������������������$�)�)�������������s�b�Z�V�e��������������������������j�Y�@�0�)�4�@�M���������������������ù������ùϹܹ�����������ܹϹùùùþ����������	����	������������������������������	���"�)�,�"��	���������������������������������������������������������������'�+�'�#�������ìçéìù������ÿùìììììììììì�T�H�;�9�5�4�4�8�;�H�T�]�a�g�k�p�m�k�a�T����� �������*�6�<�G�F�C�6�*���čĆĉĈčĚĦĪĳĹĶĳĦĚčččččč�����������������������!���������H�@�B�H�L�T�T�a�j�m�t�z�}�z�y�m�i�a�T�H�����������������������������������������'���������������'�*�3�4�6�4�3�'������'�4�M�f������t�j�]�Y�M�@�4��������������ʼ����	������ʼ�����ĳīĦĜĘĖėĚĳĿ����������������Ŀĳ����������� �(�5�6�<�7�7�.�'������������������������������������)��������)�6�B�O�t�z�h�a�[�O�B�)����ĿĳİĳĿ�������������������������̿Ŀ��������������Ŀѿֿݿ�ݿۿѿĿĿĿľ��������������������ʾ׾���Ҿʾ������=�4�=�I�I�U�V�b�e�h�b�V�O�I�=�=�=�=�=�=ƳƧƎƇƁ�|ƁƚƧƳ������������������Ƴ���ݿӿҿڿݿ��������%������E�E�E�E�E�E�E�E�E�FFF$F2F=FCF<F1F!E�E��Z�N�A�5�&����"�*�5�A�N�Y�a�i�u�s�g�Z�I�C�?�=�<�=�I�V�b�o�z�{�{�{�o�o�b�V�I�I������������$�%�$�!����������������L�G�L�L�Y�e�f�m�e�Y�L�L�L�L�L�L�L�L�L�LED�D�D�D�D�EEEEE"E*E7E9E@E=E-E*EE�� ���������
������������S�R�S�U�`�f�l�y����y�q�l�`�S�S�S�S�S�Sùïìèìôù������������������ùùùù�/�/�#���
���#�/�<�>�>�<�;�0�/�/�/�/��ܻлʻû��������ûлܻܻ������� ;  U ' L H \ 1 . 9 L N S b ? * j ) & / n ' Q B a T � k I g �  1 ` N @ " E O 4 h 2 V $ J A 9 6 > L M K f M 5 � K 9 X ( ` � g ( 4 B . r Q ! P K A , ^�J  h  J  B  �  @  �  �  �  *  �  f  Z  R  W  v  �  �  �  �  h  H  �  h    6    R  �  r  �  �  �  m  �  �  �  |  (  ,    �  �  �  �  A  �  �  �  D  ?  x  d  -  �  /  �  h  c    �  p  �  d  �  �  �    �  !  �  8  �  �  �  �  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �    9  Q  _  f  j  j  d  T  6    �  �  r  !  �  U  �  �  �  w  g  [  P  E  :  -  !    �  �  �  �  �  n  G     �  X  �  �  �  �  �  �  �  �  �  �    >  �  z  �  O  �      �  �  �          �  �  �  �  �  �  `  A  "     �  �  v  �  �  �  �  �  �  �  �  �  �  �  �  �  |  i  R  ;    �  �  �  �  �  �  ~  j  V  A  ,    �  �  �  �  �  g  B  
  �                        	      �  �  �  �  �  �  �  �  -  )  %          �  �  �  �  �  �  �  �    r  d  V  H  ~  �  �  �  �  �  �  �  �  o  Z  B  (    �  �  �  d  E  )  �  �  �  �  �  �  �  �  {  n  _  Q  C  6  (    
  �  �  �  V  �  �  �  �  �  �  �  �  �  �  �  �  �  p  <  �  {  �   �  �  �  �  �  �  |  j  V  C  0      �  �  �  s  >  	      �  �  	@  	�  	�  
  
0  
<  
<  
6  
#  	�  	�  	-  �    V  l  C  �  �  �  |  w  �  |  g  O  4    �  �  �  z  �  O  �  �    m  �  �  �  �  �  �  �  r  S  0    �  �  �  �  �  M  �  �    �  ~  g  M  D  :    ,  &  
  �  �  �  �  r  ?    �  �    L  �  �  �  �      �  �  �  �  �  P  �  �    r  �  �  �  C  3  $      %  ,  ,  &      �  �  �  i  :    �  �  A  �  �      #  &  %      �  �  �  �  t  @    �  H  �   �  �  �  �  �  {  u  o  h  a  Y  Q  J  B  :  0  '      
     �  �  �  �  �  �  �  �  �  �  �  �  �  �  }  o  Z  C  +    <  7  2  -  '          �  �  �  �  �  �  �  t  O  *    5  c  f  `  M  ,    �  �  P    �  �  �  q  2  �  �  .    W  a  a  \  U  L  ?  3  *    �  �  �  �  j  A    �  \     �  �  �  �  �  �  �  q  H    �  �  f  %  �  �  Z    �  �  _  E  .  ;  D  ;  3  ,  #      �  �  �  �  �  s  Z  @  %  ^  k  b  M  O  g  ~  x  O    �  �  �  �  �  V  5  �  _  �  �  �  �  �  �  �  �  �  }  v  n  f  ]  R  B  2  !    �  �  �  �  �  �  �  ~  a  B  #        �  �  �  �  �  �  �  o  �  �  y  e  P  1    �  �  �  N  ;  N  O  +    �  �  |  N           �  �  �  �  �  �  �  �  �  �  �  �  �  �    r  �    t  i  c  \  T  L  D  7  )      �  �  �  �  �  V    �  �  �  �  |  i  V  C  /      �  �  �  �  �  �  �  �  #    $  <  A  ;  2  %       �  �  �  x  E  �  n  /  �  �  �  e  x  }    �  �  ~  v  i  X  D  ,    �  �  �  i  .  �  �    +  :  A  K  S  [  _  ]  Y  S  G  6    �  �  �  v  N  6  �  �  }  o  X  F  `  j  c  S  :    �  �  �  e  I    �  #  
  
�    <  \  K    
�  
Q  	�  	k  �  �    y  �  �      ;  Z  ]  b  �  �  �    
    �  �  �  �    d  �  �  6  �   �  V  ^  e  m  u  x  p  g  ^  U  L  C  9  /  %      �  �  �  ,  .  /  *  $      �  �  �  �  �  �  �  �  w  j  ]  M  >  :      �  �  �  z  V  1  �  �  �  p  M  "  �  �  y  4   �      '  1  :  D  E  B  9  -      �  �  �  i  :  
  �  &                  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  r  H    �  �  e    �  �  �  �  O  I  A  7  )      �  �  �  �  q  Q  ,    �  �  �  [  3      �  �  �  �  �  �  �  m  U  :    �  �  �  �  a  "  �  [  V  M  B  3  #       �  �  �  �  �  i  C    �  �  l  -  �  �  �  �  ~  w  o  h  ]  R  E  7  (       �  �  �  w  <  �  �  �  �  �  �  �  �  �  t  e  V  G  8  *        �  �  �            �  �  �  �  �  V  6    �  w  "  �  N  j  �  �  �  �  �  �  �  ^  9    �  �  �  [  C    �  �  �  �      �  �  �  �  v  D    �  �  `    �  �  g    �  �  =  �    �  �  �  �  �  �  �  �  X    �  y    �    f  �    Y  ;      �  �  �  �  �  �    V  (  �  o      �  �  x    �  �  �  �  �  �  �  �  �  l  Y  E  3  &      �  �  �  �  �  |  d  _  L    �  �  U    �  l    �    e  �    �  K  C  2      �  �  �  �  �  v  U  3    �  �  �  V  �    ]  `  _  [  S  I  >  /      �  �  �  r  ;    �  �  i  4  \  Y  U  T  T  Q  N  A  0      �  �  �  �  l  !  �  �  u  �  �  �  �  �  �  �       �  �  �  Y  3    �  �    2  O  �  �  �  �  l  M  &  �  �  q    �  �  �  E  �  |    �    �  �  u  d  I  +    �  �  �  �  p  F    �  �  C  �  �  �  �  x  \  G  ?  9    �  �  �  �  �  Y  #  �  �  s  9  A  �    �  �  �  �  �  �  �  u  d  O  5    �  �  �  q  3  �  �  �  �  �  �  �  �  �  x  U  *  �  �  �  I  �  �    s  �  7  �  �  ~  w  h  X  I  @  :  4  (      �  �  �  �  �  k  C  j  f  b  ^  Z  V  R  P  O  N  N  M  L  U  q  �  �  �  �  �  �  	  	   	  	  �  �  �  u  9  �  �  D  �  1  �  �    �    4  /  +  '  #          �  �  �  �  �  �  �  �  �  }  n  7  8  :  <  A  G  C  9  /       �  �  �  �  e  D  &    �  }  d  J  -    �  �  �  �  t  S  0    �  �  �  �  �  �  {  8  R  `  ]  W  P  I  >  1    �  �  �  w  G    �  �  r  5  \  ;    �  �  �  �  _  8    �  �  �  Z  3      �  �  �