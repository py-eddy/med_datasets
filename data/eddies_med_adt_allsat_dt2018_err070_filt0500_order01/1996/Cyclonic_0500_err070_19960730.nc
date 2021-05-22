CDF       
      obs    R   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?�p��
=q     H  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       N   max       P���     H  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       �hs   max       <�h     H   <   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?\(�\   max       @F�ffffg     �  !�   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��(�\    max       @v}\(�     �  .T   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @/         max       @Q            �  ;$   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @���         H  ;�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ���   max       <�9X     H  =   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A��   max       B5�     H  >X   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�}   max       B4��     H  ?�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >�(   max       C���     H  @�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >D�   max       C��F     H  B0   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          t     H  Cx   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          9     H  D�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          7     H  F   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       N   max       PX��     H  GP   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��`A�7L   max       ?���!�.I     H  H�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       �hs   max       <�`B     H  I�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?\(�\   max       @F�ffffg     �  K(   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��(�\    max       @v}\(�     �  W�   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @(         max       @Q@           �  d�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @��@         H  el   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         Bt   max         Bt     H  f�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?���҈�   max       ?��(���     �  g�               (   .         s      /      
                                                )      "                                       8   =            0         
         #   	                  "                                                	      	   "   
O'0N�M�N&�NgүOݺ
O��iN���Ni�P���NPeƠNt�N{ �Oxj�O@h�N���N��O2�O�C�NzOO��Nh��N��=O� �N�{�Oj��N�4�N�F:O��NArO��N�M�O��NpLoO �]O��OR(yO�SO/v�O�0!O��O?�O	ZdPBڹPvSHOd��N�-&O/sOꨎN��OL�BO
��O*(DN�%�P�&N��@N��9OGn�N�C�O+��O��IO��N@tOpO{��O(O1�vN��ROrϾO3�pNi�O�`KN��)NތJN��`OsܺOI�gN5��O@$N?IPOG�N#��<�h<�`B<��
<���<t�<o;�`B;ě�;��
;��
;D��;o��o���
��`B�t��#�
�49X�u��C���t����㼛�㼬1�ě��ě����ͼ�����������������`B��`B��`B��`B��h��h��h��h�����������o�o�C��\)�\)�\)�\)��w��w�#�
�#�
�'',1�49X�8Q�@��@��H�9�H�9�H�9�L�ͽT���T���Y��Y��]/�e`B�e`B��%��\)��\)��hs�������w�� Ž�Q�hs).3:86.)')-5@BGGHB55-)''''''�������������������������������}��������)6BOYW]_^[B6)!
�
#/<FJNPJ</#
����������������������������������������������6@OVVO6����Z[hilkh[Z[][ZZZZZZZZ*Bg���������gN5)���������������������������������������������������������

������lmxz{����}zwqmmkllll������������������������������������������������������������26>BFOXZPOKB662222225:B[^aht�������t[OB5��������������������RUbmnq{}{znbUKRRRRRR����� ����������#$/::1/#36C\u�����|unh\NC;5355BNZ[_[NFB510555555����������������������������������������Y[^ginjg[XVUYYYYYYYY$17CJO[`hyytphd[O6($FHNSTTX^THB@@CFFFFFFRTamnxz|~���zma`ZTSR#/6:<=<<;5/#xz���������������{yx�������������������������������������������������������������������������&+=BN[t������tg[N5)&��������������������]amz����������zmaZY]���������������������������.*������������53352)������]bdn{��������{snbVX]!#./15;;0/-#!!!!!!��������������������DJUan�������znaUQHDD��������������������GIUbnz}}��}{nl_XUPIGegnt��������tpgfcbee��������������������"##+/;:940/##"""""""�"B[t����t[OB)����������������������q{{���������{sqqqqqq�������������������
����������������������������������������������������������������KTWamibaTMKKKKKKKKKKsz{�����������zmnoss5BFIGB5)##0<INQU_`UNI<80.,$#�����������������������������������������������������������.4<INUZ_^_^[USMIG;..��


�����������������������������������������������������������������������������������������������������������������nt������������yujhin��

����������?EN[gkt��}zxutg[NB@?��������������������-/6<HMSUTPJH<5/*(*--�����������������������������������ɺֺ��������ֺɺ����I�B�=�7�=�>�I�V�b�h�b�b�V�N�I�I�I�I�I�I��߼�����������������������u�{��������������������������m�`�T�C�6�,�%�&�;�T�y���������������y�m������ŹŲūŮŹ����������������������Ҿ������������������ʾξվԾʾƾ��������������������������������������������������f�M�?�,�%�$�*�4�M��������ʼݼ�μ���f��پ���������������������N�A�5�"���������5�A�Z���������v�b�V�N�t�t�o�s�t�y�g�_�Z�N�E�B�N�Q�Z�g�s�u�}�s�g�g�g�g�g�g�?�4�.�&� �!�(�0�4�<�A�M�Z�f�m�l�]�Z�B�?�Z�R�N�F�C�M�Z�g�s�����������������s�g�Zìèàßàìù����������������ùìììì�����������������
��#�%�#���
�������нʽĽĽĽ˽нݽ��� ��	������ݽнм4�0�*�+�.�4�:�@�Z�f�r�~������r�f�Y�M�4�a�]�U�O�U�Y�a�n�z�}�z�y�n�m�a�a�a�a�a�a�<�/�&���
�������������/�H�P�I�B�A�<�������������ȾǾ����������������������������������������ļʼѼռӼʼ�����������������������	�"�;�H�X�a�i�a�T�G�;�/�������������������������������������������.�+�1�3�8�;�G�T�^�[�`�f�m�y�y�m�`�T�;�.�a�`�Y�\�a�c�m�s�s�z����z�m�a�a�a�a�a�a�����������������
�� �#�&�#����
�����������������������$�0�:�=�7�0�$�����G�E�G�O�T�`�m�n�w�m�`�T�G�G�G�G�G�G�G�G�ܹϹù������������ùϹع����������������������������������������������������������������	���"�&�'�+�/�/�"��	���6�*�)�0�6�B�F�O�[�\�[�O�L�H�B�9�6�6�6�6��ܻлȻŻлܻ��������������������������������������������������àÝÓÎÇ�~Á�~ÇÓàëìù����üùìà�����y�w�t�q�y�����������������������������ݿѿĿ��Ŀ˿ѿؿݿ������$����	�������������	���(�.�0�,�*� ��	����߾�����	��"�'�"����	�������ƺưƮƯƢƢƧƩƳ���������������������h�d�c�h�u�vƁƎƎƚƝƧƩƪƧƙƎƁ�u�h�Y�Q�Y�h�x������������ּ������r�Y������:�G�������̽׽ؽͽ����l�G�!����x�s�f�Y�N�Q�Z�g�s���������������������<�8�/�(�/�<�H�U�a�b�a�]�U�H�<�<�<�<�<�<�T�M�H�B�;�4�5�;�C�H�T�a�t�z�~�z�s�m�a�T�ù����������������Ϲ�����������Ϲ�FFFFFFF$F1F=F=FBF=F<F1F$FFFFF�-�,�&�$�%�-�:�F�S�_�x�������l�_�Y�F�:�-��ƹƳƩƳƳ�������������������������������������������	���"�*�,�"���	������E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E��Z�P�S�Z�g�k�f�h�������������������~�g�Z�6�3�)�'�'�)�6�B�L�N�N�B�6�6�6�6�6�6�6�6�����ݽ׽ӽݽ�������	��������������A�4�4�0�+�+�4�A�M�Z�f�s�s�|�s�m�f�Z�M�A�����������������������������������������	�����'�,�3�@�F�L�^�b�Y�L�3�'�����������ĺɺֺ�����������ɺ����_�S�F�"�-�:�E�S�_�l�x���������������x�_����������!������������ݿҿѿȿͿѿֿݿ�������������ݿ��z�n�f�b�n�zÇÓàâåèéìîìëàÓ�z�лϻлػٻܻ���������������ܻ��a�`�b�f�n�{ņŇŌŔŚşŪŤŠŔŇ�{�n�a����ŹŶŲŷŹ�������������������������ƻ����x�l�d�_�_�c�l�x�����������������������������������Ľнݽ��������ݽĽ���²¦±²¿��������¿²²²²²²²²²²�{�o�b�V�I�4�2�3�=�I�X�b�o�{ǆǍǔǔǋ�{����������$�*�&�/�$�����������������������}�|���������������������*� �&�*�6�C�N�O�P�O�C�6�*�*�*�*�*�*�*�*������Ľ�������������
������
������ĦĠĜĕďĚĦĳľĿ��������������ĿĳĦE*E(E)E*E.E7ECEGEEECE7E7E*E*E*E*E*E*E*E*��
�����������������
��#�%�%�(�(�%�#��a�a�T�R�H�D�?�H�T�\�a�d�a�a�a�a�a�a�a�aD�D�D�D�D�D�D�EEEE*E-E.E*E'EEED�D����������������������������������������� . @ 8 D / # & E " ` < H � $ : t Q ' " = X 8 \ H 0 a T 3 ` S ; v H v B R h 4 @ e 5 _ J i J 7 A  % = Z M ' m w : V & � a S < G 0 - S 8 5 h l D ? E , , 4 > f R R 1 Y  "  �  D  �  �  !     {  �    �  �  �  �  �  �    1    �  �  }  �  {  �  \  �  	  >  y    �  \  �  s  %    ,  �  �  M  �  C  r  F  �  �  x    �  �  A  h  �  3  �  �  �  
  �  U  �  \  ?  	  O  �  �  Z  �  s  !  �  �  �  �  �  v  �  i  I  c<e`B<�9X<e`B<T�������%   %   ��G�;D���0 żt��#�
�u�\)�u��j��j�0 Ž+�C���9X��j�,1��`B�'��o��+���q���+�C��\)�ixս0 ŽL�ͽ',1�Y��'<j��P�� Ž�vɽP�`�<j�H�9����P�`�P�`�D���}�ixս����L�ͽaG��q���D���aG����-���
�P�`�u��%�aG���+�ixս�O߽�t����-������%�������㽾vɽ�-��1��Q�\���m���B�3BgB�*B(�B��B.B5�B��B�RB��B	�_BwPB�VBK�B~mA��?B��B!�RB!��B]�B�B"\CB'�B�}BԃB1��B��B�cB�>B�qB��A��A�b�B�B�B��B�(B*RB)�vB�AB%SA�5BQTB-:,B��B(�B�B�9B��B��B'�`B	��B��BN;Bl�B��B)]vB��B#I�B"ߪB�9B��A��B I�Bz�B&Y�B�B�B!�pB&�}B`eB��B@UBlhB
��B�B
��B�3B��BO�B�B�B�BBB��BB�.B?�B4��B?/B�xB��B
9�B��B�$B��BD�A���BBdB!�QB!|�BD�B:@B"uB'�mB�&B~B1��B�|BE�B�9B��B0�A�}A��B��B=B��B FB)��B*>2B	&B@�A��B@rB-(�B��B)9�B!�B9�B��B�$B(*�B	��B>BA�B��BY�B)��B?�B#�NB"�B�MB+qA���B P7B��B&?�B�B
�B"+>B&��B@�B�B�lB@6B
�%B�TB
??B�BͫBAB��B��@<~�B�A��@��Aj]A���ANM�Ar��@��AV�&A��:A��*A���A<!�A���A�JA��jA,�O@٪2A��<A�G]AM �@�g.A�}�A��tAd>�A�w�A�P�B	$HAhV�>���A��-A��Aؿ�@�{A�F7A�hAp��A�,[AZĨAY��BDtB�@�J4A��A��A��A���>�(C���@�O�B��A���C�(�A���A��A-�uA=�qA��f?�U�@Ah`@��A�xOA~��A��t@���A�1�A�ġ@��UA&σA�f�B]EB	^OA�6�B `�A�-�A��C��(A�'�A�q�C�^�AJ&@C�BN�A @�k�Aj��A�][AN	�At��@��AW'A�o�A�}xA��6A<]�A���A͇A��A,��@ױ�AƄ7A�tQAM9�@�:�A�wA�v(AgA���A�pB	I�Ah�>���A���A��A׈�@���A�	�Ä�Ap�]A��iAZٽAZ�B`nB��A�}A�_A�̜A��;A�p�>D�C��F@|��B��A�HC�&�A�pGA؀�A,�WA=��A���?���@@l�@�+YA��kA~��A�4@�XA�pA���@��A$�A�}B AB	@�A���B �.A�f�A�`C���A���A��C�_�AIF�               (   /         t      /      
                                                )      "                                       8   >            1         
         $   
                  "                                                
      
   "   
               %            9      1                              #         !                                    '            !            9   3            %                  -                                                                                                            -      +                                                                                                   7   1            %                  -                                                                                 N��N�M�N&�NgүOQ >Oj�'N���Ni�P<.NP&8:Nt�NN�Oxj�O�N���N��N��bO��SNG�O�[Nh��N��=Orf�N�{�Oj��N�4�N�F:O^NArOa=N�M�O��NpLoO
*�O��O2Z�N�^wO/v�O��/O��O֪O	ZdP6�/PX��O0d�N=��N��OꨎN��OL�BO
��OjhN�%�P�&N��@N7trOGn�N�C�O+��O��eO��^N@tOpO{��O(O"rN��ROrϾO3�pNi�O�`KN��)NތJN��`OsܺOI�gN5��O@$N?IPOCN#��  �  �  �  �    �  �  �  	%  J  �  h  �  �  h  <  �    G  �  �  v  �    �  r  N  �  �  V  ^  @  �  �  �    �  (    �    v  $  i  �  �    {  �  �  �  �  s  �  �  E  +    �  
  !  �  �  e  �  E  �  L  C  �  �  �  o  f  �    �  !  +    �  �<���<�`B<��
<��㻣�
;D��;�`B;ě���/;��
�ě�;o�o���
�e`B�t��#�
�D����o���㼣�
���㼛�㼼j�ě��ě����ͼ�����P������h��`B��`B��`B�����o����h�+���+���+��w�\)�t����\)�\)�\)��w�#�
�#�
�#�
�'49X�,1�49X�8Q�D���T���H�9�H�9�H�9�L�ͽY��T���Y��Y��]/�e`B�e`B��%��\)��\)��hs�������w�� Ž�vɾhs#)/6664)'')-5@BGGHB55-)''''''�������������������������������}��������&)68BKMPPMFB6)$#/<CGKLHD</#���������������������������������������������)4=>)������Z[hilkh[Z[][ZZZZZZZZ39Bgt�����������NB53���������������������������������������������������������

�������lmxz{����}zwqmmkllll������������������������������������������������������������46ABCOWXOB9644444444=B[bekt�������}taO==��������������������RUbmnq{}{znbUKRRRRRR����� ���������#$/::1/#36C\u�����|unh\NC;5355BNZ[_[NFB510555555����������������������������������������Y[^ginjg[XVUYYYYYYYY'49GO[hwuplh]ZOB6+)'FHNSTTX^THB@@CFFFFFFRTamnxz|~���zma`ZTSR#/6:<=<<;5/#zz}�������������|zzz��������������������������������������������������������������������������3BGNt������tg[N@5-+3��������������������_almz������zmba[[__���������������������������-'�����������/101-)������bcgq{����������{nibb"#/199/#""""""""""��������������������DJUan�������znaUQHDD��������������������GIUbnz}}��}{nl_XUPIGegnt��������tpgfcbee��������������������"##+/;:940/##"""""""�"B[t����t[OB)����������������������z{����������{zzzzzz�������������������
����������������������������������������������������������������KTWamibaTMKKKKKKKKKKsz{�����������zmnoss5BFIGB5)##0<INQU_`UNI<80.,$#�����������������������������������������������������������.4<INUZ_^_^[USMIG;..��


�����������������������������������������������������������������������������������������������������������������nt������������yujhin��

����������?EN[gkt��}zxutg[NB@?��������������������./8<HKQSSNHF<9/,)+..���������������������ֺպɺ��������ɺֺֺֺ�������ֺ��I�B�=�7�=�>�I�V�b�h�b�b�V�N�I�I�I�I�I�I��߼�����������������������u�{��������������������������`�Y�G�C�:�;�B�T�`�m�y�������������y�m�`����ŹŵŮűŹ�������������������������ƾ������������������ʾξվԾʾƾ��������������������������������������������������Y�@�7�1�.�/�6�@�Y�r���������¼�����r�Y��پ���������������������N�5�)�������(�5�A�Z�i�t�|�x�q�Z�N�t�t�o�s�t�y�g�`�Z�N�F�H�N�O�Z�g�p�s�|�s�g�g�g�g�g�g�?�4�.�&� �!�(�0�4�<�A�M�Z�f�m�l�]�Z�B�?�Z�Y�N�J�I�N�V�Z�g�s�w�����������w�s�g�Zìèàßàìù����������������ùìììì�����������������
��#�%�#���
�������н̽Žͽнݽ�������
�����ݽнннм4�1�+�,�/�4�;�@�^�f�m�r�|���~�q�f�Y�M�4�a�`�U�S�U�[�a�n�y�w�n�e�a�a�a�a�a�a�a�a�/�)���
���������
��#�/�C�H�N�H�?�<�/�������������ȾǾ����������������������������������������ļʼѼռӼʼ�������������������	��"�/�;�H�R�a�e�a�T�C�;�/������������������������������������������.�+�1�3�8�;�G�T�^�[�`�f�m�y�y�m�`�T�;�.�a�`�Y�\�a�c�m�s�s�z����z�m�a�a�a�a�a�a�����������������
�� �#�&�#����
������	���������������$�0�4�7�0�/�$���G�E�G�O�T�`�m�n�w�m�`�T�G�G�G�G�G�G�G�G�ܹϹù��������ùϹܹ������������������������������������������������������������������	���"�&�'�+�/�/�"��	���6�*�)�0�6�B�F�O�[�\�[�O�L�H�B�9�6�6�6�6���ܻͻλлܻ��������������������������������������������������àßØÓÑÇÁÃÃÇÓàçìøýÿùìà�����y�y�w�t�y�����������������������������ݿѿĿ��Ŀ˿ѿؿݿ������$����������������	��#�*�,�)�%�"���	������߾�����	��"�'�"����	�������ƽƳƲƱƳƳ���������������������������h�d�c�h�u�vƁƎƎƚƝƧƩƪƧƙƎƁ�u�h�r�f�`�k�z�����������
��ּ������r�!���
��:�S�l�������˽νƽ����c�G�:�!�����s�g�]�R�V�Z�g�s���������������������<�;�8�<�H�U�`�[�U�H�<�<�<�<�<�<�<�<�<�<�T�I�H�=�;�9�;�H�J�T�[�a�m�o�z�z�m�l�a�T�ù����������������Ϲ�����������Ϲ�FFFFFFF$F1F=F=FBF=F<F1F$FFFFF�-�,�&�$�%�-�:�F�S�_�x�������l�_�Y�F�:�-��ƹƳƩƳƳ���������������������������������������������	���"�'�%�"���	����E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E��Z�P�S�Z�g�k�f�h�������������������~�g�Z�6�3�)�'�'�)�6�B�L�N�N�B�6�6�6�6�6�6�6�6������ݽֽܽݽ������������������������A�4�4�0�+�+�4�A�M�Z�f�s�s�|�s�m�f�Z�M�A�����������������������������������������	�����'�,�3�@�F�L�^�b�Y�L�3�'�������������ɺֺ����������ֺɺ����l�S�F�=�B�F�J�S�_�l�x���������������x�l����������!������������ݿҿѿȿͿѿֿݿ�������������ݿ��z�n�f�b�n�zÇÓàâåèéìîìëàÓ�z�лϻлػٻܻ���������������ܻ��n�b�a�h�n�t�{ŇŔřŝŠŨšŠŔŐŇ�{�n����ŹŶŲŷŹ�������������������������ƻ����x�l�d�_�_�c�l�x�����������������������������������Ľнݽ��������ݽĽ���²¦±²¿��������¿²²²²²²²²²²�{�o�b�V�I�4�2�3�=�I�X�b�o�{ǆǍǔǔǋ�{����������$�*�&�/�$�����������������������}�|���������������������*� �&�*�6�C�N�O�P�O�C�6�*�*�*�*�*�*�*�*������Ľ�������������
������
������ĦĠĜĕďĚĦĳľĿ��������������ĿĳĦE*E(E)E*E.E7ECEGEEECE7E7E*E*E*E*E*E*E*E*��
�����������������
��#�%�%�(�(�%�#��a�a�T�R�H�D�?�H�T�\�a�d�a�a�a�a�a�a�a�aD�D�D�D�D�D�D�EEEE*E*E,E*E$EEED�D����������������������������������������� & @ 8 D &  & E " ` 9 H � $ ( t Q & # > W 8 \ = 0 a T 3 5 S > v H v A C n ; @ ] 5  J a K , * + % = Z M " m w : d & � a P ( G 0 - S 8 5 h l D ? E , , 4 > f R R 0 Y  �  �  D  �  �  �     {      �  �  �  �    �        m  :  }  �  �  �  \  �  	  5  y  �  �  \  �  ,  �  �  �  �  �  M  2  C    �  �  T      �  �  A  J  �  3  �  a  �  
  �  0    \  ?  	  O  W  �  Z  �  s  !  �  �  �  �  �  v  �  i  %  c  Bt  Bt  Bt  Bt  Bt  Bt  Bt  Bt  Bt  Bt  Bt  Bt  Bt  Bt  Bt  Bt  Bt  Bt  Bt  Bt  Bt  Bt  Bt  Bt  Bt  Bt  Bt  Bt  Bt  Bt  Bt  Bt  Bt  Bt  Bt  Bt  Bt  Bt  Bt  Bt  Bt  Bt  Bt  Bt  Bt  Bt  Bt  Bt  Bt  Bt  Bt  Bt  Bt  Bt  Bt  Bt  Bt  Bt  Bt  Bt  Bt  Bt  Bt  Bt  Bt  Bt  Bt  Bt  Bt  Bt  Bt  Bt  Bt  Bt  Bt  Bt  Bt  Bt  Bt  Bt  Bt  Bt  _  �  �  �  �  �  �  �  Z  .    �  �  w  E  !  �  �  �  b  �  �  �  w  h  Y  J  ;  ,    	  �  �  �  �  {  X  /     �  �    t  j  \  M  >  (    �  �  �  �  ^  +  �  �    <   �  �  �  �  �  �  �  �  �  �  �  �  �  �  u  g  X  J  <  .     �    4  M  b  r  |  ~  r  ^  @    �  �  �  {  9  �    j  �  �  �  �  �  �  �  �  �  u  ;  �  �  f    �  1  �  3  �  �  �  �  �  �  �  �  �  �  �  �  �    o  ^  H  3       �  �  �  �  �  �  �  �  �  |  m  \  L  ;  $    �  �  �  �  �  q  �  �  �  	  	"  	  	  �  �  G  �  y  M    �    O  �  �  J  G  E  B  @  =  ;  8  6  3  ,  !          �   �   �   �   �  u  �  �  �  �  �  �  �  �  v  \  ?    �  �  n    �    �  h  V  D  0      �  �  �  t  C    �  �  2  �  A  �  >   �  �  �  �  �  �  �  �  �    '            �  �  �    =  �  }  t  i  \  N  =  )    �  �  �  �  �  �  l  W  B      �  �    @  T  b  e  [  G  '  �  �  Y  �  �    �  6  �  �  �  <  2  '         �  �  �  �  �  |  b  H  -    �  �  �  �  �  �  �  �  �  v  S  1    �  �  �  �  w  W  5    �  �  �            �  �  �  �  �  �  �  d  =    �  �  r  ;    B  F  B  :  0  $    �  �  �  �  �  �  \    �  �  D  �  �  H  f  {  �  �  �  �  �  �  �  �  �  �  �  h  '  �  �  9  �  �  �  �  �  �  �  �  r  M  )    �    	  �  �  �  m  *   �  v  u  t  s  r  q  p  o  n  m  k  j  i  i  l  o  r  u  x  {  �  �  x  n  e  [  O  C  7  +        �  �  �  �  �  �  �  �  �      �  �  �  �  �  �  �  �  �  �  f  F    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  {  r  h  ^  r  q  R  *  	  �  �  �  �    _  @    	  �  �  �  W     �  N  C  8  -  ,  -  .  .  /  0  .  )  $      �  �  �  �  �  �  �  �  �  �  �  �  �  �  ~  l  W  A  1  *  #          Q  �  W  �  �  �  �  �  c  9    �  g    �  \  �  w  �  S  V  Y  \  _  b  d  d  d  d  d  c  `  ]  [  X  U  R  P  M  J    H  \  Q  @  0  "                 �  �  �  Z  �  �  @  .    
  �  �  �  �  �  y  ]  @  #     �   �   �   �   x   ]  �  �  �  x  f  S  @  5  .  '      �  �  �  �  �  �  �  e  �  �  q  b  W  L  :  "    �  �  �  �  �  g  >    �  �  �  �  �  �  �  �  e  B    �  �  m  $  �  �  3  �  p     s  �  �  	      �  �  �  �  �  �  �  �  �  y  ^  H  $  �  �    �  �  �  �  �  �  �  �  f  7     �  �  f  *  �  �    �  8    !  $  '  '  $         �  �  �  �  �  �  �  �  �    x    �  �  �  �  �  �  �  �  �  �  t  `  E  !  �  �  �  ~  ^  a  �  �  �  �  i  K  )    �  �  �  �  u  J  	  �  T  �  N            �  �  �  �  �  �  �  e  A    �  �  �  a  ,      S  m  Y  A  #    �  �  �  h  8    �  �  Y  #  �  �  $  !          �  �  �  �  �  �  �  �    q  d  [  R  J  L  `  @  "      �  �    �  �  �  r  &  �  Q  �  0  v  Z  �  �  �  �  �  �  �  x  G  �  �  (  �  :  �  j  �    �  �  �  �  �  �  �  �  �  �  �  �  _  7    �  �  E     �  [   �  �  �            �  �  �  �  �  �  �  y  g  U  D  O  l  w  y  z  {  {  w  o  d  R  8    �  �  �  �  W  J  �  ~   �  �  �  �  m  M  '  �  �  �  ~  N    �  f    �    T  <  ?  �  �  �  �  �  x  j  X  D  (    �  �  [    �  C  �  6  �  �  �  �  �  k  M  *    �  �  {  I    �  �  {  ;  �  �   �  �  �  �  �  �  �  y  e  N  5    �  �  �  �  j  B    �  �  q  p  [  =      �  �  �  �  �  �  �  �  e    �  f    �  �  �  �  �  p  J    �  �  C  �  �  P  �  �  O  �  �  W    �  m  V  8    �  �  �  �  x  P    �  �  �  �    �  �  Q  E  >  8  /  '        �  �  �  �  �  �  �  �  �  �  �            *  �  �  �  �  w  \  <    �  �  �  G    �  �    �  �  �  �  �  �  o  W  ?  %    �  �  �  x  o  N  ,    �  �  �  �  �  y  k  ]  O  A  -    �  �  �  �  �  �  |  l  
    �  �  �  �  �  �  �  �  �  r  T  0  !          �      
  �  �  �  �  i  <    �  �    E  �  �  �  j  �  O  [  �  �  �  �  �  e  D  !  �  �  �  t  4  �  �  [    �     �  �  �  �  �  �  �  �  �  �  ~  {  x  u  r  n  k  h  e  a  e  X  K  <  ,      �  �  �  �  �  �  x  W  /    �  ^   �  �  �  �  �  �  w  ^  D  ,      �  �  �  �  �  o  P  @  K  E  <  3  )      �  �  �  �  �  �  �  g  J  -     �   �   �  �  �  �  l  R  :  #      �  �  �  �  r  Q  B    �  �  y  L  G  C  >  9  /  &        �  �  �  �  �  �  �  j  B    C  7  &            �  �  �  �  �  {  W  2    �  �  U  �  �  �  �  h  L  1    �  �  �  �  `  %  �  �  o  %  �  �  �  �  �  �  �  �  �  �  �  �  �  �  i  M  %  �  �  b    �  �  �  �  v  Y  8    �  �  �  t  H  !    �  �  �  o  1  �  o  k  h  a  T  H  ;  /  #  (  /  2    
  �  �  o  ?    �  f  V  F  5  %      �  �  �  �  �  n  U  C  1        �  �  �  �  �  �  �  �  �  �  �  �  t  h  _  X  R  J  >  1  $    
    �  �  �  �  �  �  �  s  K    �  �  C  �  �  t  �  �  �  �  �  r  \  S  8  :  7  /  #      �  \    �  Y   �  !    �  �  �  �  �  S    �  �  �  Q    �  �  v  :  �  �  +    �  �  �  �  q  K  $  �  �  �  p  =    �  �  d  �  U    �  �  �  �  t  Q  -  	  �  �  �  [  %  �  �  t  1  �  �  �  �  �  �  �  �  n  U  5    �  �  B  �  p  �  |  �  v  �  �  �  q  V  =  $  
  �  �  �  �  �  x  ]  <    �  �  U  �