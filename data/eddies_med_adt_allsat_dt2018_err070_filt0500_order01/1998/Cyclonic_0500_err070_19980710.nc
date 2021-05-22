CDF       
      obs    I   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?����l�     $  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M��   max       P��     $  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ��^5   max       =t�     $  �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?��R   max       @E]p��
>     h  !   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ���
=p�    max       @vzz�G�     h  ,�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @1         max       @Q�           �  7�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @�v�         $  8|   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ���#   max       <T��     $  9�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�7z   max       B*�Y     $  :�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�|Q   max       B*�z     $  ;�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       =��   max       C��(     $  =   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       =��   max       C���     $  >0   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          K     $  ?T   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          E     $  @x   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          =     $  A�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M��   max       P���     $  B�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��u��"    max       ?��1���.     $  C�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ��^5   max       =o     $  E   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?(�\)   max       @E]p��
>     h  F,   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ���
=p�    max       @vzz�G�     h  Q�   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @0�        max       @Q�           �  \�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @���         $  ]�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         E:   max         E:     $  ^�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?���n   max       ?���C,�     �  _�            $      
   	   	   2            >                  =   "                     %            )   %         !                                     &      ;   ?                     J   "   
            	   '      #         0            O #�N���Ns	�O�k"N��5N�-�N(ڮO"^oP8��N\4dN��wO���O�C�NynYOhC�M��NC��N�]�P��Pn��NS]�O�9O� N-z�O�j�NN��P��NbB�NG�N��GO��PH�N
1�O�!PE�Pf�Nh�`N�nOL~O�@�NO<JOo~0OU�O�
YN�G�P�pO{oN�BOį{P#Q�O|�N��N�PN5��P2�OPoO�$�O��wO̌N�BO��OJ��O��O���O #O闍NwN�N���O���Nx;�O'cN�H�O$��=t�<�C�<�o<49X;��
;��
%   %   ��o�D����o�o�o�o�o�t��49X�D���e`B�u��o��C���t���t����㼛�㼼j�ě��ě����ͼ�������������`B��h�����o�+�C��C��\)�t��t���P��w��w��w�',1�,1�49X�8Q�<j�@��@��D���D���L�ͽT���q���q���q���}�}󶽇+��7L��hs������P���-��^5��^5������������������ABHO[hnmlhg[OIB@AAAA���

������������#'5<>6/#
�������NN[glpnhg[VNMLNNNNNN;<EHUUailaUH><;;;;;;#)///;<3/##��������������������t������������tmloot
#,(#"
��������������������)25AO_aWOI6)&+36BO]``dih[OB6*"!&mt��������trlmmmmmmu����������������}wu

166?BOXZOBB611111111��������������������fo���� 	�������tgf<P[an�����������nU8<��������������������?BCKN[gt{����tg[WNB?��)6[ht[O6)����}���������vy}}}}}}}}.2;@HNRT\]]YVTH;/#+.lnwz������znllllllll6;Tam��������maTH956`ahlnqz����zna``````������������������������������������������������������������������"������mnz������znmmmmmmmm_bfn{�������{nb`cb`_TXam�����������zmaUT��������������������^amoonoma\[Y^^^^^^^^##%08<IRRNI?<0#"#&##��������� ����������15BJNQ[gxutg[>0,,21��������������������������
 
�������������������������������������������������������������������8BNg����������tgN<480<IUY`^[ZYUPI>0*&&*0:<=BIJU[abbcbaUIB<;:fglt�����������tmkf���:BIRW^YOB)������
#03:;6#
����hht�����~tnhhhhhhhhhotw���������troooooo��������������������N[g����������t[NGEEN`gjtx�����������tea`����������������������� $��������rz������������zvrsrrKOT[hmsrklihf[VOMHKKY[gt�����������tic[Y�������������������� "!�����
#<HKK=?</)
���npu{������������zonn������
����������<<>HUWUOKH<348<<<<<<{�����������{{{{{{{{���!"������������������������������ ����������##/26;<BED<4/*#!#HHUannuyxuongaUSLJGH�ʼƼּ̼ټ����������������ּʿѿοĿ������Ŀѿݿ�������ݿѿѿѿѼY�S�R�M�K�M�Y�f�r�z�x�r�h�f�Y�Y�Y�Y�Y�YÆÀÄ�~ÇÓàì��������������ùìàÓÆ�������������ĿѿҿؿѿȿĿ��������������A�?�4�3�4�4�:�A�M�U�W�S�M�E�A�A�A�A�A�A�t�m�g�[�[�[�g�t�t�t�t�t�t�t�t�t�Ľ��Ľнؽݽ��������������ݽľ����|�p�s�z���������ʾ�����	��׾�����������}�����������������������������ìèìùþ��������������������ùìììì��	�������	���"�/�;�@�H�X�]�[�T�H�/��F�:�-�+�(�/�F�S�_�x�������������x�l�_�F�"������"�'�/�;�;�>�;�/�"�"�"�"�"�"�Y�P�O�S�Y�f�r�������������������r�f�Y�����������żʼμʼ����������������������/�#�#�����#�*�/�/�5�/�/�/�/�/�/�/�/�<�:�3�<�H�K�U�a�g�n�p�v�n�a�U�H�<�<�<�<�N�5�!���N��������� �����������|�g�N�~�Z�5�*�"�%�5�\�g�s�������������������~����������������������������������������	��������������	��������	�p�o�v�s�m�p���������������������������p�g�`�Z�Y�Z�g�s�z�������s�g�g�g�g�g�g�g�g���������j�s�����������������������������T�R�H�E�@�C�H�T�X�\�\�X�T�T�T�T�T�T�T�T��������������5�J�S�X�M�5�������������������������������������������y�u�n�y�������������y�y�y�y�y�y�y�y�y�y��޿ݿҿݿ��������������������������������$�0�6�;�:�2�0�$������T�G�.����޾��"�.�G�y�����������m�T����������������	�����������������������N�H�A�7�/�)�5�A�N�Z�b�g�s�w�u�s�g�c�Z�N��ƳƤƕƁ�u�\�R�\�uƎ�����������������̺ɺ��̺ϺȺ������ֺ���,�4�U�h�_�!��ݺ��H�A�D�H�T�a�m�w�s�m�a�T�H�H�H�H�H�H�H�H����������������������������������������������������������%�*�9�A�?�6�*���������������/�C�O�\�`�]�Q�C�6�*�������������ɺֺݺٺֺɺ������������������������������������Ŀѿݿ�����ݿѿĿ�����������������������(�7�)������뻅�x�j�k�x�������ûлԻ̻û»������������@�<�?�@�G�L�Y�e�m�r�e�Y�T�L�@�@�@�@�@�@���������������������������%�'�'���꽄�{�|���������нݽ�����нý����������4�)�(�������"�(�4�=�A�M�Q�M�D�A�4�h�[�O�B�6�.�,�,�/�6�B�O�n�~ĎĊčā�t�hÇ�z�q�k�e�a�[�aÇàù������������ùàÇ�ڻѻĻ��������������ûлܻ�������ݻں�������#�'�*�'���������ŔŊŇ�{�y�w�{ŇŔŠŧŬţŠŔŔŔŔŔŔ������������������������¦¦������������������¿¦�/�)�#�������#�/�<�H�W�^�U�M�H�<�/��ʻ������ûܻ���4�7�6�3�)�!������?�6�8�@�M�Y�f�r��������������r�f�Y�M�?�������������������
��
�	�����������̹ù����������ùϹܹ����������ܹϹù�����ķĹ�����������������
��������������������������������
�����
�������ؾ(�"�����(�4�A�M�S�Z�b�_�Z�M�A�4�(�(ED�D�D�D�D�D�EEE7EPE\EhE\ETEGE7E*EE�лû������������ûлܻ���������ܻջ�����������$�0�=�H�I�5�0�(�������E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E������������Ŀѿ׿տѿƿĿ����������������.�!����!�.�:�S�y�������������l�S�:�.���������������������ʾ׾׾׾Ѿʾ����������������������������	��"�2�5�"�	������E�E�E�E�E�E�F FFF$F+F$F"FFFE�E�E�Eٹ����������������ùϹܹݹܹ۹ӹϹù����� 8 % 8 " ( O = U 6 [ R D 9 ^ K 5 q P U 4 D C r U n h A X Q i + L � @ d j b G - n < + G ? S = J 2 2 0 N w 7 F # . : = Q 7 S ) H < = u > d V O � i 4    Z  �  �  �  �  �  O  w  A  r  �  P  e  �  �    �  �  &  �  u  t  W  9  �  �  w  �  (  �    �  �  >  �  +  �    �  �  m  �  �  %  �  �  9    �  �    {  �  i  �  �  +  `  Z    '  �  +    %  �  �  �    �  �  4  m<t�<49X<T������%   ���
�t��t��P�`�t��D�����
��\)�#�
���D���e`B��t����P�D����j�+��P��j�@���1�u����������/��7L��%��h�C��}�}��P���@��@��'ixսixս�7L�H�9�y�#���-�e`B���ͽ�
=�u�H�9�L�ͽL�ͽ��㽇+���#��u����������
��C����ͽ�\)���ͽ����-���ٽ�����`B��G�B�\B�B#��B	�B�+B�5B1gB"�B��B$ƻB �Bj�BHDB�PB�8B$s(BJ�B�B�RBc�BS�B	B�lBuA�7zB�'A�i�B2B*�YB*�B��Bp�B��B(�A��*B��A��B&$@BN~B��B!d~B�Bu�B ��B�2B	UB&��B'�B)�Bb�B$�Bw�B
U�B!�B	�oB
j�B1�B{B .dBRB
:RB|�B?/BǓB)$B<�B�BG�B�$BI�BYvBA�B1�BCnB�wB#��B?�B�CB��B<�B"֋B�B$�,B �BC�B@B��B�5B$r1BYsB<5BCB��BE"B	@�B�&B��A�|QB�A�}�B��B*�zB*+B��B��BB(�`A���B��A���B&$�By�B�B!?�BĦB@HB BB�rB	3�B&I�B',GB@QB;]B$�MBC@B
@�B ֑B	��B
��B>BV�B >-B@^B
oOB��BB�B�B)B�B�uB�+B�B�nB?�B�qB��B?�A�A{�;@���A��AAw��A;��A�);A-�APbAGɵA�vWA� @��HA���@��@��dA���Aň3A�bA��tA��dAY�/A�FPA�5KA�=�A�LsA�+�A�:Ao�A��#B	�Ad�,A�нA��[B�_@K!A�
A�{EA��A��f@2�Ay��A�D�@��e?՟�A�5A$&[A7��A��tA�;�@�I�?���A�R�?!�xA�_�A�J�@��>@�e�A�>}zTA��A�SA:�C�|�@�j�B	�}C��Ax��A�AN�A�a�C��(=��A
�A|�@�-fÀ�AwȘA;�	A�~RA.�AP�8AGDAAΖ�A��C@�@gA���@�!�@���A��A�yA�o A�(A�|RAZ%�A�:A�{rA�y�A�T�A�}�A�7An�_A�lYB	>A`��A�{�A�a�B>H@K�A���A�n�A��A� B@/$�Ay�A�S@�é?�x9A��2A! �A8WA��Aʄ^@�0�?��A��E?/A���Aà�@���@��A�>���A��A�~A:��C���@��dB	zCC��Ax�+AAN�QA�g�C���=��            %         
   
   3      	      ?                  =   "                     &            *   &         "   !                                  '      <   @                     K   #               
   '   	   #         1   	                     !               /         !                     E   7         '            %               1         )   5            !                  )         !   )               %      %                     %      '         '                                       %                              =   -                     #               1            !            !                                             #                           %      '                     O�"N���Ns	�O��N��5N�-�N(ڮO"^oO��N\4dN��wO���Oi�NynYOQ7M��NC��N�]�P���P6�8NS]�Ok�OO��N-z�O[_VNN��O�>�NbB�NG�N��GOV�PH�N
1�O�!O�V�O���Nh�`N�nO:�EO�@�NO<JOo~0OU�O,�cNM��O�{�Oq N���OS�O�4�O|�N��N�PN5��Pz�N��Og�@O��wN��N8j-O��OJ��O��O���O #O闍NwN�N+�oO�%NNx;�O'cN�H�O$��  q  e    �  y  �  �  �  �  I  B  �  	n    �  I  f    C  "    m  w  w  �  G  K    _  V  �  �  �  �  P  <    �    �  P  7  z  �  �  �  �  �  �  �  �  `  �      w  	G  5     �     K  Q  p  M  �  �  �  k  �  /  �  �=o<�C�<�o;D��;��
;��
%   %   �u�D����o�t����o�#�
�t��49X�D����9X���
��o����ě���t����ͼ����/�ě��ě����ͽo����������`B��P��P���o�C��C��C��\)�t��0 Ž�w�8Q�#�
�#�
�q���}�,1�49X�8Q�<j�D���Y�����D���P�`�u�q���q���q���}�}󶽇+��7L�������-���P���-��^5��^5��������������������ABHO[hnmlhg[OIB@AAAA���

�������������#&/4850)
����NN[glpnhg[VNMLNNNNNN;<EHUUailaUH><;;;;;;#)///;<3/##��������������������tuuy��������������tt
#,(#"
��������������������)-2=BOZROG6))-26BORVXYXPOB@64,))mt��������trlmmmmmmw����������������yw

166?BOXZOBB611111111��������������������n�������������tknS_nz����������znaLCS��������������������MN[gty����tg[NDFMMMM	')6BGW^][OB6)	}���������vy}}}}}}}}.15;@HOTYXWTRH;/-,,.lnwz������znllllllll9ATamz������}maTH>99`ahlnqz����zna``````������������������������������������������������������������������"������mnz������znmmmmmmmm_bfn{�������{nb`cb`_^fm����������zmaZYZ^��������������������^amoonoma\[Y^^^^^^^^##%08<IRRNI?<0#"#&##�������� �����������15BJNQ[gxutg[>0,,21��������������������������
 
�������������������������������������������������������������������BN[g��������tgNC=<=B0<IUX_^ZYXURI?0*''*0:<=CILUZ`bcb`UQID<<:sw~���������������ts��)5=DHHB5)!������
#03:;6#
����hht�����~tnhhhhhhhhhotw���������troooooo��������������������O[g����������t[NHFFOntu������������vtlnn����������������������� $��������tz���������zwsssttttOO[hmkh[ONOOOOOOOOOOY[gt�����������tic[Y�������������������� "!�����
#<HKK=?</)
���npu{������������zonn������
����������<<>HUWUOKH<348<<<<<<����������������������� �������������������������������� ����������##/26;<BED<4/*#!#HHUannuyxuongaUSLJGH�ּмʼɼмּݼ������	��������ֿѿοĿ������Ŀѿݿ�������ݿѿѿѿѼY�S�R�M�K�M�Y�f�r�z�x�r�h�f�Y�Y�Y�Y�Y�YàÓÇËÐÓÚàìù��������������ûìà�������������ĿѿҿؿѿȿĿ��������������A�?�4�3�4�4�:�A�M�U�W�S�M�E�A�A�A�A�A�A�t�m�g�[�[�[�g�t�t�t�t�t�t�t�t�t�Ľ��Ľнؽݽ��������������ݽľ�׾��������������������ʾ׾��������㾌������}�����������������������������ìèìùþ��������������������ùìììì���������	���"�/�;�=�H�V�\�X�T�H�/����_�S�F�@�<�B�F�S�_�l�x���������z�x�l�_�_�"������"�'�/�;�;�>�;�/�"�"�"�"�"�"�Y�R�P�T�Y�f�r�������������������r�f�Y�����������żʼμʼ����������������������/�#�#�����#�*�/�/�5�/�/�/�/�/�/�/�/�<�:�3�<�H�K�U�a�g�n�p�v�n�a�U�H�<�<�<�<�A�-���"�N�\���������������������q�Z�A�Z�5�'�'�+�4�A�N�s���������������������Z��������������������������������������������������	�	�����	������������s�r�w�}�����������������������������g�`�Z�Y�Z�g�s�z�������s�g�g�g�g�g�g�g�g�����������������������������������������T�R�H�E�@�C�H�T�X�\�\�X�T�T�T�T�T�T�T�T��������������5�G�O�Q�D�5�������������������������������������������y�u�n�y�������������y�y�y�y�y�y�y�y�y�y��޿ݿҿݿ��������������������������������$�0�3�8�7�0�-�$������T�G�.����޾��"�.�G�y�����������m�T����������������	�����������������������N�H�A�7�/�)�5�A�N�Z�b�g�s�w�u�s�g�c�Z�NƳƧƝƌƉƎƗƧƳ��������������������Ƴ��Ժɺ����������ɺֺ�����%�&�-�!����H�A�D�H�T�a�m�w�s�m�a�T�H�H�H�H�H�H�H�H����������������������������������������������������� ���#�*�6�8�?�;�6�*���������������/�C�O�\�`�]�Q�C�6�*�������������ɺֺݺٺֺɺ������������������������������������Ŀѿݿ�����ݿѿĿ�����������������������(�7�)������뻅�x�v�p�x�x�����������������������������L�A�I�L�Y�e�h�m�e�Y�L�L�L�L�L�L�L�L�L�L�������������������������������ؽ��|�}���������нݽ���߽н½����������4�+�(�������(�4�:�A�L�M�P�M�B�A�4�[�O�B�@�6�6�4�4�6�9�B�O�^�n�u�w�t�q�h�[Ç�z�u�r�p�q�u�zÓìù��������üôàÓÇ�ڻѻĻ��������������ûлܻ�������ݻں�������#�'�*�'���������ŔŊŇ�{�y�w�{ŇŔŠŧŬţŠŔŔŔŔŔŔ������������������������¦¦¿�����������������¿¦�/�$�#���#�/�;�<�H�I�U�X�U�N�H�F�<�/�/��޻Իӻ߻�������$�"���������?�6�8�@�M�Y�f�r��������������r�f�Y�M�?��������������������� ���������������̹ù¹����ùϹչ۹ѹϹùùùùùùùùù�����ķĹ�����������������
��������������������������������
�����
�������ؾ(�"�����(�4�A�M�S�Z�b�_�Z�M�A�4�(�(ED�D�D�D�D�D�EEE7EPE\EhE\ETEGE7E*EE�лû������������ûлܻ���������ܻջ�����������$�0�=�H�I�5�0�(�������E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E������������ĿѿտѿѿĿ������������������.�!����.�:�S�o�y�����������l�`�G�:�.���������������������ʾ׾׾׾Ѿʾ����������������������������	��"�2�5�"�	������E�E�E�E�E�E�F FFF$F+F$F"FFFE�E�E�Eٹ����������������ùϹܹݹܹ۹ӹϹù����� 5 % 8 ) ( O = U - [ R @ ( ^ K 5 q P ^ * D C O U _ h @ X Q i ! L � @ B C b G * n < + G # 2 7 K / 2 - N w 7 F ' * % = T & S ) H < = u > ^ B O � i 4      �  �    �  �  O  w  �  r  �    H  �  �    �  �  4    u  =  �  9    �  %  �  (  �  �  �  �  >  v  �  �    �  �  m  �  �  r  c  �  -  �  �  �    {  �  i  �  �  �  `  !  R  '  �  +    %  �  �  j  n  �  �  4  m  E:  E:  E:  E:  E:  E:  E:  E:  E:  E:  E:  E:  E:  E:  E:  E:  E:  E:  E:  E:  E:  E:  E:  E:  E:  E:  E:  E:  E:  E:  E:  E:  E:  E:  E:  E:  E:  E:  E:  E:  E:  E:  E:  E:  E:  E:  E:  E:  E:  E:  E:  E:  E:  E:  E:  E:  E:  E:  E:  E:  E:  E:  E:  E:  E:  E:  E:  E:  E:  E:  E:  E:  E:  ^  k  p  p  k  b  S  ?  "  �  �  �  `     �  �  K  �  �  �  e  a  ^  Z  S  J  B  ;  5  /  $      �  �  �  �  �  K      
          �  �  �  �  �  �  �  �  �  �  �  �  �  w  �  �  �  �  �  �  �  �  �  �  �  �  f  /  �  �  P    B  �  y  w  u  r  o  g  `  X  O  D  9  -  !      �  �  �  �  �  �  �  �  q  `  L  8  $    �  �  �  �  �  �  �  h  O    �  �  �  �  �  �  �  o  M  %  �  �  �  ]  )  �  �  �  N    �  �  �  �  �  �  �  �  p  ]  G  1      �  �  �  �  �  �  c  c  �  �  �  �  �  �  �  �  �  �  ^  +  �  �  5  �  �    9  I  D  ?  :  2  )           �  �  �  �  �  �  �  �  �  �  B  3  $      �  �  �  �  �  �  �  �  �  �  w  k  O    �  �  �  �  �  |  o  _  O  A  <  /       �  �  �  U  "   �   �     T  �  �  	  	=  	`  	m  	d  	F  	  �  �  -  �  -  ?  �  (   �           �   �   �   �   �   �   �   �   �   �   �   �   �   z   n   b  �  �  �  �  �  �  �  |  _  =    �  �  _    �  +  �  �  +  I  A  9  1  )           �  �  �  �  �  �  �  �  �  �  �  f  `  Y  S  L  E  ?  8  2  ,  &          �   �   �   �   �   �    �  �  �  �  �  �  �  �  �  �  t  h  Y  E  1      �  �    +  ?  B  9     �  �  z  1  �  �  �  �  k  <  �  �  m    	              �  �  �  �  {  ^  G  (  �  �  y    �  9        �  �  �  �  �  �  �  �  �  �  �  �  �  t  L    �  a  h  l  k  e  ]  S  F  6  %    �  �  �  �  t  Q  0    �          $  @  w  r  d  S  ;    �  �  �  Q  �  �  k  .  w  p  j  c  \  S  K  B  8  .  $      �  �  �  �  �  k  Q  �  �  �  �  �  �  �  �  �  �  M    �  �  l  :  �  �  l  �  G  F  E  D  C  B  A  @  ?  >  ?  B  E  H  K  N  Q  T  W  Z  =  D  J  G  <  )      �  �  �  �  �  Y    �  X  �        |  z  w  u  r  o  m  j  h  f  f  e  e  d  d  d  c  c  b  _  \  Z  W  T  R  O  L  J  G  @  5  )         �   �   �   �  V  Q  M  H  C  >  9  5  0  +  &                �   �   �  �  �  �  �  �  �  �  �  �  U    �  i  �  �    �  �  �  �  �  �  �  �  �  �  u  ^  ?    �  �  �  r  �  f  	  �    ^  �  �  �  �  �  �  �  �  �  �  �  �  �  �  |  r  g  \  Q  F  �  �  �  �  �  �  �  �  x  j  Y  F  4       �   �   �   �   �  �  �    (  N  N  =  "       �  �  �  k    �  a  �  m  f  '  
    %  9  .    �  �  �  �  �  m  O  (  �  �  B  �  �        �  �  �  �  �  �  �  �  �  v  b  M  8  #    �  �  �  �  �  �  �  �  �  �  {  q  g  ]  S  H  ;  .        �   �  	          �  �  �  �  �  m  F    �  �  �  N    �  r  �  �  �  |  g  Y  P  F  (    "    �  �  �  �  h    �  �  P  C  6  )         �  �  �  �  �  �  �  �  �  �  �  �  �  7  .  "    �  �  �  �  �  �  n  V  7    �  �  H    D  /  z  e  Q  >  ,      �  �  �  �  g  :    �  r    �  (  �  �  �  �  �  �  �  �  �  �  �  r  +  �  �  4  �  s  }  �  9  �  �  �  �  �  �  p  ^  L  @  2  #    �  w  2  �  �  w  3  i  �  �  �  �  �  �  �  �  �  �  h  D  !  �  �  �  S  
  �  z  �  w  s  |  s  a  F  (    �  �  r  $  �  q    y  �  e  �  �  �  �  �  �  �  y  [  =    �  �  �  r  @  	  �  �  h    Z  �  �  �  �  �  �  �  �  E  �  �    z  �  �  �  �  �  Z  �  �  �  �  �  �  �  �  �  \    �  �  8  �  A  }  W   �  �  �  �  �  �  �  �  �  f  J  0    �  �  �  �  �  x  �  �  `  \  Y  V  V  a  m  x  }  x  r  m  a  R  B  3      �  �  �  �  �  |  t  i  ^  T  I  ?  5  +      �  �  �  �  �  �       �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �            �  �  �  �  u  \  C  (    �  �  �  �  �  ]  �  �  �    5  R  d  q  w  t  f  S  7    �  �  �  [  !  �  �  B  k  �  �  	  	*  	;  	E  	E  	;  	+  	  �  �  '  �    ;  0  �  �  5  *      �  �  �  n  A    �  �  d    �  �  n  5  �  ]  �  �  �  �  �  �  �  k  J  (  
  �  �  �  �  �  1  �  @   �  5  3  A  |  �  �  �  �  �  �  �  �  �  y  ]  ?    �  �  �               �  �  �  �  �  �  �  z  Z  1  �  �  i  �  K  E  ?  :  6  1  )      �  �  �  j  E  �  �  0  �  U   �  Q  G  <  1  &      �  �  �  �  x  Z  7    �  �  �  q  V  p  ^  K  0    �  �  c    �  l  /  o  �  �  �  N  �  ^  �  M  C  9  .  #      �  �  �  �  �  �  �  x  M     �   �   l  �  �  ]  7    �  �  �  �  ~  �  n  7  �  �  C  �  �  �  g  �  �  }  a  C  $    �  �  �  }  ]  =  !    �  �  �  �  �  �  �  �  �  �  �      "  $  !      �  �  �  �  �  �  �  �  [  i  b  V  C     �  �  ~  ?  �  �  m  !  �  :  �  �    �  o  [  G  3      �  �  �  �  p  P  1    �  �  v  (   �  /    �  �  �    �  d  G    �  �  w  :  �  �  u  .  �   �  �  �  �  l  I  $  �  �  �  l  4  �  �  \  "  /    �  w  �  �  �    b  B  !  �  �  �  �  l  C    �  �  U  �  z     ~