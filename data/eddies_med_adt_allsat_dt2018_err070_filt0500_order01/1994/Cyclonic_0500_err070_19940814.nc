CDF       
      obs    H   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?pbM���   max       ?�$�/��        �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       NI	   max       P�
T        �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ���#   max       <u        �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?=p��
>   max       @F\(��     @  !   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��Q��    max       @v|��
=p     @  ,L   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @)         max       @R@           �  7�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�~        max       @�x�            8   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �gl�   max       :�o        9<   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�DU   max       B5&]        :\   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A��e   max       B4��        ;|   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >dL�   max       C��F        <�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >Bi�   max       C��=        =�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          �        >�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          ?        ?�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          3        A   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       NI	   max       P��        B<   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�N;�5�Y   max       ?��,<���        C\   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ���#   max       <T��        D|   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?=p��
>   max       @F\(��     @  E�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�z�G�     max       @v|��
=p     @  P�   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @)         max       @Q�           �  \   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�~        max       @�L�            \�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         ?�   max         ?�        ]�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�t�j~��   max       ?���`A�8     �  ^�                     7                                                                 '      I            
      !         
         g      Q                                    �   ,   4   	               	   	   	                  
N� �N�XBO�'�N}{Nj�uPQ�-O�$NQ|�N�7�Oxr�N�?�NCrN�k�O��aO���N�&�O���O�>wO�MO��N�hTO[ZN�_WO��O/QP��N���O���O|aWP#!�OfANj��N&��N���P-OXuNI	Ns��N���N��aNtN4P�
TN�AP]�jN���O��cN6YN���OD�LN�v�N?�TOU��N��O%EBNU~�OɚPS�O��<N�4�O_�)OlS�N�~�N͊1O�EN�[sNw�>Nm�O�x0O�N�#N�ްN){<u<T��<T��;o:�o%`  �D����o��o��o�ě���`B��`B�o�t��t��t��#�
�49X�49X�D���e`B�u��o��o��C���C����㼛�㼛�㼣�
��1��j��j�ě��ě����ͼ��ͼ�/��h��h��h��h��h�o�t��t����#�
�',1�0 Ž8Q�H�9�P�`�Y��e`B�m�h�m�h�y�#��%��+��+��+��\)��hs��t������j���#�!'&���������FHIT^akmnmieba^TJHEFntz���������������nn��������������������������������������������$'#!���������������������������������������������W[\glgggtv|trg[ZTUWW��������������������;<IUbnzynmbUUUI><:;;t��������xtttttttttt����������������������������������efm|�����������ztmae������������������������*7FHNIB=)����������������������LP[gt��������urupfNL��������������������./<HPSPJHE<61/......s{����������{vvvtrss;BFLNU[ghmge[ONB;;;;bt������������tjcacb56@COU\]bb\OHC64//15wz����������������zw����������������������������������

 ��������
&,<JU]gjpgZR0
��
()6BOS[ZPOBA6)#��������������������


� #/<HSUWULH<3/($#"  �������������������������������������uz�����zwsuuuuuuuuuu��� ��������������������������������:;HKORQQKHG;:546::::TTagmvz��zomcdaTTTTTs|�������	
�����|vs����������������������������-66/����#037910.#�����������������������������������ghtt���������tljhegg���������������|{|}�@BNZ[bghlge[NB?;@@@@56BEIMGB876555555555�������

����������������LOU[hmsysljh][OLHEHL�����������������������
#&%"
���������������������������������������������~����������������|~~��������������������5BNg��������tgYN@;55mnz}������zungcemmmm��������������������/05<@ISU^afb\VIC<0'/������������������������������������������������������������)5BNVOOHFB)���glot����~~~utg^[]cgSUacghihaXURQQSSSSSSzz������������~zzzz@HUabcaUIH@@@@@@@@@@�ֺӺѺҺֺ�������������ֺֺֺֺֺ�������ƯƧƤƧƮƳƴ���������������������	�������������������	����"�%�+�"��	�!�� �!�#�-�:�?�F�P�F�B�:�-�!�!�!�!�!�!�����������������������������������������#�/�X�h�l�s�����������������������s�A�#�$�������������������$�0�?�A�@�:�0�$�|�u ¡���������������&�(�+�)�(�����M�J�Q�S�M�Z�e�s���������������s�f�Z�M���������������ʼռּ�����ּҼʼ���²©²¸¿��������¿³²²²²²²²²²�������������������������������������������������������������������������������������ݿۿ׿ֿݿ�������&�(�.�*������ھؾ������	�
��	�������𾥾�����t�l�`�Z�f�s��������������������u�h�P�O�C�I�M�O�\�h�uƁƈƧƷƼƧƚƁ�u�	����������	��4�G�T�^�T�G�;�.�"��	�ֻĻܻ�����'�@�M�Y�g�f�Y�M�'������Z�Q�S�S�Z�g�s�������|�s�g�_�Z�Z�Z�Z�Z�Z�����������Ľнݽ����������ݽнĽ��������������������������������������������ѿпҿ������(�5�:�9�5�(�#����ݿѾ��׾ԾѾ׾�����	������	���������������ĽĺĸĿ������� �0�<�?�<����;�2�1�/�6�;�H�I�O�N�J�H�;�;�;�;�;�;�;�;���������׾����	����	�����׾ʾ����)���� ��������+�6�;�H�P�X�O�6�,�)�a�N������������A�Z�f�s����������a�a�]�U�T�I�J�Q�U�a�n�zÂ�{�z�v�z�n�e�a�a����������������������������������������ŹŹŹ������������������ŹŹŹŹŹŹŹŹ�U�O�I�K�M�U�X�a�d�n�p�s�z�~�z�n�j�a�U�U�����������)�E�]�W�[�b�b�f�[�B�5�������ݽнͽʽнݽ����,�<�1�(����������#�)�6�:�8�6�)�����������g�]�\�g�n�s�����������s�g�g�g�g�g�g�g�g���������ĿĿѿܿݿ�ݿٿѿĿ����������������������������������������������������	���	�
���"�/�7�/�-�"���	�	�	�	�	�V�=�$������������=�V�l�wǂ�ǈ�{�o�V�"��"�%�/�8�;�H�H�H�;�/�"�"�"�"�"�"�"�"�������r�f�p�y�������!�/�:�>�6���ʼ�������������������������ŭŧŠŝŔŃŅŌŔŠŭŹ������ſŽ����ŭ�����������������������迆���y�y�x�y�����������������������������ѿοſ����ĿͿѿݿ������"�������������*�6�6�C�D�C�=�=�9�6�*�����_�[�[�_�l�x�������x�s�l�_�_�_�_�_�_�_�_������������������������)�+�+�'� ��T�T�L�T�T�_�`�k�m�y�|�y�v�u�o�m�h�`�T�T���������������ùϹܹ��������ܹϹù��O�C�I�O�[�h�i�h�d�[�O�O�O�O�O�O�O�O�O�OD�D�D�D�D�D�D�D�EEE,E6E+EEED�D�D�D��0�������������#�<�U�b�n�k�o�n�d�U�I�0������������'�*�4�4�1�'������S�L�S�X�_�l�x�����������������x�l�_�S�SĦĥęčā�t�qĆĚĦĳĺĿ��������ĿĳĦ�W�X�`�l�tāčĚĝġĚěĘĒčć�t�h�\�W�Y�X�W�Y�e�j�r�~�����������~�r�e�Y�Y�Y�Y�����������������������º������������������������������~�����������������������������������������������������������������������������!�$�'�'�!��	�����û��������������û˻лڻڻлûûûûû�Ç�\�X�W�S�M�U�a�nÇÓÞ����øôìàÓÇ��
�����������
��#�/�<�H�I�K�H�<�/�#���ؾ޾����	��!���	���������������z�u�w�z����������������������������F$F!FF"F$F1F5F;F1F,F$F$F$F$F$F$F$F$F$F$ # _ @ + O R ) ^ J k Q ] G 6 \ E 2 ; G u 7 * t T % B d R L ` L / J ? l ` H , ; _ � 7 G _ 7 Q 5  ] T T F t 6 ; . ) . n R � + B X + Q ] L � h > Y  �  -  ~  �  �  �  	  ]  �  �  D  V    �  k  �  f  �  �  �  �  G  �    =  �  �  4    ^  P  |  \    �  �  .  �  �    �  8  ;  �  �  3  [  �  �  �  d  �    j  e  �  �      �  c  �    =  �  �  �    �  �    q:�o:�o��`B�o���
���m�h�t���`B�e`B����T����t���󶼬1�u��P�+�o�0 ż��
�t���9X�'��D����9X�m�h�t���j�<j������/�+�P�`�ixռ�����P�t��o�+�+��;d���T���,1�0 Žm�h�8Q�T���e`B�L�ͽ����m�h�gl�������G���+���P��E����T������P���w����������G��\��-�����+Bb�A��-B��B,2�B5&]B�B�B��B	-tB�8B'[vBԭB�sB!}B x�B�B�{B�B	��B w�B�SB)j�B�+B
޴B0�B�BP�B9�B�{B&(CB7�B!OiB�B��Br�B!`A��B#�B(A�DUA��B�B ϟB-�B%i�B�B! �B�B* pBA�B�B*�B��BW�BB�B0NB7B)aBN�B	�.B�B�{B&�DB8�B!��B��B�dB	��BOB#�B�>BJ�A�|�B� B,>�B4��B�zBR�B�IB	>�B7iB'@�B��B�?B<�B ��B(B��B?�B	ȶB H~B6mB)�#B�
B
D�B0�gB:hB�pB@ B8B&��B?�B!B�B5�B?�B�eB!@A��B�B@�A��eA��BBVB ��B,�B%��BMNB �EBw�B)��BBCBF�B�dB�>BC�BB9�B?hB@OB)=fB�B	��BĪB07B&��B`B!� B�B��B	��B�B��B�@G)qBFA�}�@wY(AJN�A�ʳB	&A��9A3T�AEQ~@�R�A�A5A��iA�X�A�swAW\�AF��B+�A^?�@ư[A���A)*�A��jA��AW��A�5�A��yAS߁A�"�A;�AƋAH�A��A�]}A��=A1��AրA�W�Az
mA���A�xB
��A�ѨAZ@��A���?��Aq
~A�PA��k@���A�@jAj&�>dL�A�*�C�@4A�S@�>@��A���Aܧ�?�S@8jA�-�AH��@b�k@�o�A�bA��<AX�gA��9C��F@D(�B5�A�8�@uO6AJ�A��B	@A��A4��AF�@���A���A��PA�L�A�YgAX�AHpBǖA[ @�՟A��9A*zA���A���AW�A�տA��AS .A�jA9�Aƀ;AH��A� �A�qA���A3iA�|A�SAzޗA���A�hB
�.A���@��t@��nA�Q?/L�Ao��A~�pA��[@��5A҃ Ai��>Bi�AڄC�9A냊@�v@�tKA��A��?��O@n-A�G�AH�@d�@��A�a�A�o+AY�:A��_C��=                     8                                       !   	                      (      J                  "                  h      Q                                    �   ,   5   	               	   	   	                           !         /   %         !                        !   #   '            #      )            3               +                     7      ?                                       '            #                     !                              /                                       #   '                  )                                                3      3                                                   !                     !            N_N�XBOV;eN}{Nj�uPQ�-OiuNQ|�N�7�OP�|N�?�NCrN�k�O9k�O���N�&�OGH�O��O�MO�:tNkb�N��nN=ܕO��Nʆ�P��N���N���O|aWO���N��Nj��N&��N���OX`GO��NI	Ns��N���N��aNtN4P��N�AP;��N���Ow�N6YN���OD�LNYg�N?�TOU��N��NO�hNU~�O;�O`h&O{��N�4�O_�)O\��N�~�N͊1O�EN�[sNw�>Nm�O�x0O�N�#N�ްN){    �  �  �  �  �  O  �  R  b  �  U  �  �    9  �  T  �  �  �    �  1    �  �  �  A  t  m  �  r  �  �  S  c  �  �  2  �  
�  U  	  o  �  �  �  �  w  d  �  '  �  �  m    	l  N  �  �  �  �  a  �  �  �  T  �  �  �  i<49X<T��;�`B;o:�o%`  ��j��o��o���
�ě���`B��`B�u�t��t���C��D���49X�D���T�����
��o���
���㼋C���C���㼛��8Q�o��1��j��j�C������ͼ��ͼ�/��h��h����h���o��P�t����#�
�,1�,1�0 Ž8Q콁%�P�`���ٽ��P�u�m�h�y�#��o��+��+��+��\)��hs��t������j���#$"FHIT^akmnmieba^TJHEFyz���������������{y��������������������������������������������$'#!���������������������������������������������W[\glgggtv|trg[ZTUWW��������������������;<IUbnzynmbUUUI><:;;t��������xtttttttttt����������������������������������efm|�����������ztmae��������������������%)+6:?@<3)���������������������LP[gt��������urupfNL��������������������2<HNROH<732222222222z{����������~}{zzzzzMNW[egjgc[NIMMMMMMMMfjt������������tqief367CKOPY\^\\QOC63233wz����������������zw����������������������������������

 ��������&+0<IU]`aa\UI<0#&66BBORQOB=6066666666��������������������


� #/<HSUWULH<3/($#"  ������	
�����������������������������uz�����zwsuuuuuuuuuu��� ��������������������������������:;HKORQQKHG;:546::::TTagmvz��zomcdaTTTTT}�������	
�����}vt}��������������������������)1/������#037910.#������� ����������������������������ghtt���������tljhegg���������������|{|}�=BDNV[_a[NBA========56BEIMGB876555555555�������

����������������NOX[hhojh[ONNNNNNNNN������������������������

���������������������������������������������~����������������|~~��������������������7BN^gt��������tgZNA7mnz}������zungcemmmm��������������������/05<@ISU^afb\VIC<0'/������������������������������������������������������������)5BNVOOHFB)���glot����~~~utg^[]cgSUacghihaXURQQSSSSSSzz������������~zzzz@HUabcaUIH@@@@@@@@@@�ֺԺպֺ������������ֺֺֺֺֺֺֺ�������ƯƧƤƧƮƳƴ��������������������������������������������	������	���!�� �!�#�-�:�?�F�P�F�B�:�-�!�!�!�!�!�!�����������������������������������������#�/�X�h�l�s�����������������������s�A�#������������������$�0�3�7�7�2�0�$��|�u ¡���������������&�(�+�)�(�����M�T�V�R�Z�f�j�s���������������s�f�Z�M���������������ʼռּ�����ּҼʼ���²©²¸¿��������¿³²²²²²²²²²�������������������������������������������������������������������������������������ݿۿ׿ֿݿ�������&�(�.�*������ھؾ������	�
��	���������t�i�f�d�f�p�s�����������������������u�h�T�P�O�T�\�h�uƁƚƧƲƹƭƧƚƎƁ�u�	����������	��4�G�T�^�T�G�;�.�"��	�ٻƻܻ�����'�4�M�Y�g�e�Y�M�'������Z�V�U�Z�g�s���z�s�g�Z�Z�Z�Z�Z�Z�Z�Z�Z�Z�Ľ½��������Ľнݽ����ݽԽнĽĽĽ��������������������������������������������ݿֿ׿߿������(�5�5�0�(����������ؾ׾־׾�������	�
���	���������������ĽĺĸĿ������� �0�<�?�<����;�2�1�/�6�;�H�I�O�N�J�H�;�;�;�;�;�;�;�;�ʾǾ��ʾʾ׾��������׾ʾʾʾʾʾ��)���� ��������+�6�;�H�P�X�O�6�,�)�A�(������(�4�A�M�U�h�u������s�Z�A�U�U�U�U�]�a�n�p�q�n�g�a�U�U�U�U�U�U�U�U����������������������������������������ŹŹŹ������������������ŹŹŹŹŹŹŹŹ�U�O�I�K�M�U�X�a�d�n�p�s�z�~�z�n�j�a�U�U��
������)�5�>�B�I�K�O�M�B�5�)��������ؽݽ������%�(�5�+�(���������#�)�6�:�8�6�)�����������g�]�\�g�n�s�����������s�g�g�g�g�g�g�g�g���������ĿĿѿܿݿ�ݿٿѿĿ����������������������������������������������������	���	�
���"�/�7�/�-�"���	�	�	�	�	�=�$������������=�V�k�vǁ�~ǈ�{�o�V�=�"��"�%�/�8�;�H�H�H�;�/�"�"�"�"�"�"�"�"���{�s�u�~�������!�.�:�8�2�����ּ���������������������������ŭŠŞŔŅņōŔŠŭŹ������ŽŻ����žŭ�����������������������迆���y�y�x�y�����������������������������ѿοſ����ĿͿѿݿ������"��������*�����"�*�6�:�;�7�6�*�*�*�*�*�*�*�*�_�[�[�_�l�x�������x�s�l�_�_�_�_�_�_�_�_������������������������)�+�+�'� ��T�T�L�T�T�_�`�k�m�y�|�y�v�u�o�m�h�`�T�T�ù����������ùϹӹڹѹϹùùùùùùù��O�C�I�O�[�h�i�h�d�[�O�O�O�O�O�O�O�O�O�OD�D�D�D�D�D�D�D�D�EEEEEEEED�D�D��#��������#�0�B�I�O�T�U�Q�I�<�0�#���������������'�)�3�3�1�'����S�L�S�X�_�l�x�����������������x�l�_�S�SĦĥęčā�t�qĆĚĦĳĺĿ��������ĿĳĦ�[�Y�a�h�l�o�t�xāčĚğęĚėĒčĆ�t�[�Y�X�W�Y�e�j�r�~�����������~�r�e�Y�Y�Y�Y�����������������������º������������������������������~�����������������������������������������������������������������������������!�$�'�'�!��	�����û��������������û˻лڻڻлûûûûû�Ç�\�X�W�S�M�U�a�nÇÓÞ����øôìàÓÇ��
�����������
��#�/�<�H�I�K�H�<�/�#���ؾ޾����	��!���	���������������z�u�w�z����������������������������F$F!FF"F$F1F5F;F1F,F$F$F$F$F$F$F$F$F$F$  _ = + O R  ^ J g Q ] G  \ E % : G w 3 - M T " B d 1 L X S / J ? > Q H , ; _ � 8 G b 7 S 5  ] / T F t + ; + $ * n R � + B X + Q ] L � h > Y  i  -  �  �  �  �  �  ]  �    D  V    �  k  �  �  k  �  �  �  �  H  �  �  �  �  �    �  J  |  \    �  R  .  �  �    �  -  ;  �  �    [  �  �  q  d  �    _  e  �  �  �    �  t  �    =  �  �  �    �  �    q  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  �  �  �  �        �  �  �  �  �  v  Z  <    �  �  �    �  �  �  �  �  �  l  H  "  �  �  �  �  h  >    �  �  a  �  q  �  �  �  �  �  �  �  �  }  g  N  0    �  �  Q  �  y    �  �  �  �  �  �  �  ~  v  n  c  U  G  :  ,          �   �  �  �  �  �      +  8  D  P  Q  G  =  7  9  ;  =  >  ?  @  �  �  �  {  k  R  1    �  �  �  k  6    �  �  v    �     |  �  �    3  B  L  O  C  .    �  �  5  �  2  y  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  t  a  M  8  "    �  R  N  K  H  E  B  ?  9  3  -  &             �   �   �   �   �  ^  `  a  Y  Q  H  ?  7  /  %      �  �  �  �  �  Z  !   �  �  �  �  �  �  �  x  k  Z  H  4      �  �  R    �  w  3  U  S  Q  P  M  J  G  D  @  =  8  2  -    �  �  �  �  �  �  �  �  �  �  �  �  w  b  K  5      �  �  �  �  �  Z  7    n  �  �  �  �  �  �  �  �  �  y  U  0  	  �  �  r    �         	    �  �  �  �  �  �    a  :    �  �  N  
   �   �  9  7  6  4  .  &          �  �  �  �  �  �  �  }  \  ;  q  �  �  �  �  �  �  �  �  �  �  �  l  0  �  �    �  �  #  B  R  R  K  E  @  6  *        �  �  �  q  %  �  �  U  _  �  |  n  d  i  b  l  v  y  }  �  s  Y  2  �  �  b    �    �  �  �  �  �  i  a  W  8    �  �  o  4    �  �  y  v     �  �  �  �  �  �  �  �  �  �  z  i  V  :    �  �  �  f    �  �  �  �          �  �  �  �  i  9    �  �  �  k  >  �  �  �  �  �  �  �  �  �  �  {  g  U  C  7  2  -  .  1  4  �    )  1  .  "    �  �  �  �  |  M  <  �  �  u  1    �  �  �    	    	    �  �  �  �  �  �  �  h  @    �  E   �  �  �  �  �  �  �  u  [  ?    �  �  �  A  �  �  �  .    W  �  �  �  �  �  ~  o  `  P  >  ,    �  �  �  �  �  �  �  �  �  8  f  �  �  �  �  �  �  �  �  �  �  n  �  ^  �  %  w  Y  A  @  =  2  !    �  �  �  �  f  O  &  �  �  �  W    �  �  1  �  �  !  K  c  p  s  i  d  f  <  �  �  B  �  �  1  �  n    D  m  �  �  )  A  W  g  i  R  4  �      �  L  �  \  �  �  �  �  �  �  �  ~  x  s  o  j  f  `  Y  S  L  D  <  4  ,  r  b  S  C  4  $      �  �  �  �  �  �  �  �  �  �  �  �  �  H    )  :  ?  A  9  1  (      
     �  �  �  �  �  Z  �  �  �  �  �  �  �  �  �  �  �  �  �  �  u  >  �  �  +  �  �    E  Q  R  K  @  5  +  4  )  �  �     l  �    a  �   �  c  L  5      �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  t  _  K  5      �  �  �  �  �  �  �  �  �  �  y  g  R  :    �  �  �  �  j  L  0       2  +  $      	     �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  
�  
�  
�  
\  
.  	�  	�  	f  	�  	�  	�  
   	�  	�  	*  H  K  N  !  �  U  H  ;  /  "    
  �  �  �  �  �  �  �  �  �  �  �  �  �  �  	  	  	  �  �  �  b  #  �  �  !  �  �  l  �  @  �  �    o  c  X  M  >  .        �  �  �  �  �  �  �  r  x  }  �  �  �  �  �  �  �  �  �  x  u  �  �  o  ^  D    �  r  8  �  �  �  �  }  y  v  r  n  j  e  b  `  ^  ^  b  f  v  �    W  �  �  �  �  �  �  �  �  �  �  �  �  }  v  n  g  ^  V  M  E  �  �  �  e  F  %    �  �  �  m  I  "  �  �  s  &  �  ?   k  h  k  n  q  t  u  p  j  e  `  S  @  -      �  �  �  �  �  d  L  4    �  �  �  G  
  �  �  b  )  �  �  w  9  �  �  u  �  �  �  �  �  �  x  _  D  '    �  �  �  {  Y  5  �  �  �  '      �  �  �  �  �  �  q  ]  I  4      �  �  �  �  �  �  �  �    :  c    �  �  �  �    _  /  �  �  I  �  }    �  �  �  �  �  �  �  {  g  V  D  3  $      �  �  �  �  �  �  O  �  �  n  �  F  i  h    �    i  �  c  t  B  �  w  	P  _  �  �  �  �  �  �  	      �  �  �    -  �  k  �  �   �  	7  	k  	b  	R  	B  	-  	  �  �  �  ]    �    �  �  @  i  Q  �  N  <  +        �  �  �  �  �  �  �  �  e  Y  T  D  "  �  �  �  �  |  t  c  K  .    �  �  �  y  a  (  �  �  4   �   _  �  �  �  �    W  .    �  �  ~  .  �  p  :  �    9  k  �  �  �  �  �  �  �  t  T  4    �  �  �  �  v  I    �  �  m  �  �  �  �  �  y  l  _  Q  ?  ,       �  �  �  �  W      �  a  \  X  P  F  <  /  #      �  �  �  �  �  �  �  g  :    �  �  �  �  �  �  �  �  �  �  w  _  G  *    �  �  �  �  }  �  �  t  g  ]  U  P  L  M  N  P  R  V  c  o  h  Z  L  <  ,  �  �  �  �  {  f  O  6      �  �  �  �  �    �  N  	   �  T  ;  #  E  -    �  �  �  �  �  `  3  �  �  n    �  �   m  �  �  �  �  �  �  r  J  &    �  �  �  o  G     �        �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  v  �  �  �  �  v  ^  C  &    �  �  �    U  &  �  �  �  ~  h  i  i  h  ^  Q  5    �  �  �  w  M  "  �  �  �  p  C  �  �