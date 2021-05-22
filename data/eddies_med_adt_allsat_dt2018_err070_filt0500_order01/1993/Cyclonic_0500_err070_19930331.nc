CDF       
      obs    O   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?�-V�     <  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M�@�   max       P��P     <  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ��v�   max       =o     <   $   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?:�G�{   max       @FTz�G�     X  !`   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��33334    max       @v\          X  -�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @,         max       @Q@           �  :   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @��          <  :�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ��   max       <�1     <  ;�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�|   max       B0�     <  =(   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A��   max       B/�     <  >d   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >�   max       C��3     <  ?�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >>�   max       C���     <  @�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          G     <  B   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          A     <  CT   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          A     <  D�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M�@�   max       P��     <  E�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�5?|�i   max       ?�1���-�     <  G   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ��v�   max       =o     <  HD   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?J=p��
   max       @F.z�G�     X  I�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��33334    max       @v\          X  U�   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @(         max       @Q@           �  b0   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @�M          <  b�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         =�   max         =�     <  d   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?���n/   max       ?�1���-�     �  eH            7            	      $            6      	   G               ,                         
               %      
      ,         '          "      
         *         0   
         *                           ,               
   &                     N�A#N���NM�;On�<O1�N�N�S[M�@�NH�{P��N@�*N(dCOI��P^Q|NN�aN�YPI�=P	�O�zN�+N$��P\w7P1��O��N�h�N���O�EN�/�N��^N�I?N�<AN^�HO�EO?ٔP�O��N���OC�2PL�"N���NLO���OY�PO���PˤNFLO(�PN��N��DOz�O��#O�VP��PN��WO�KN[JsO��NK�OA�N͍?O���NI�YN��O�v,N:jO�r�O?N���O��M�M�hYO��3NY/pO�*�O1�/O��@O,M�Nl!�N��j=o<�j<���<u;ě�;ě�;�o;�o;�o;D���D���ě���`B��`B�o�#�
�#�
�49X�D���D���D���e`B�e`B��C���t���t���t����㼣�
��9X��9X��j��j��j��j��j�ě��ě��ě����ͼ�/��h��h��h��h��h��h��h��h�����+�+�+��P��P������w��w��w�'49X�<j�@��L�ͽT���T���]/�]/�ixսq���q���u�u�u�y�#��vɽ�v���������������������!),5;BLN[\[NB5-)&!!!knz�����ztnjkkkkkkkk���)231,)�����������������.56?BNOUZYQNB55.....")6BDFBB6)(""""""""=BCOOQOBA7==========���������������������������������

���������������

������������&)5BEFINRZNB50w�������������{rpegw����������{������������������������������������������������r{�����������������rST]adeimpssmcaUPPPQS�����������|}��������������������:H\mz����y~�~maTH43:�#0IUn||yp<0#����������������)))5BKJHB5*)���������������������������������������#/<HH<3/#��������������������GINSUbdkllfb\UNIGGGGu{~������������{{tuu��������������������������
#$%#
����������������������������#
��������#0<IMTTSNI<10'#!�����znea[^anz������;BKNV[dpmhf_[NB@966;���������#("���������������������������inz����znciiiiiiiiii*6CIVajqr\C*@FN[gtx�������tgNB?@}��������������vtux}����������������������������������������������

��������$$#�))/6<<;6.)!��� -10-)��������������������������"&(&!������$6Og[6���������ntx��������{thnnnnnn�����������������/5;BGNSQNB53////////y��������������|yxy������������������������������������������������������������r�����������������yr{{{���������{x{{{{{����

��������Z[gt������������th\Z����� ������������� :BO[hosph[OB6)+5BHNUVSNNDB5.-,,)++��������������������������������������������������������������������������������9<HUanz����{znaUHB=9xzz��������zxxxxxxxx����������������������&%"�������<IUcjswwrnbIHHF>;;:<
#0<CIUZUI<4#
	
��


���������������


 ������#�����#�'�/�<�?�H�R�H�>�<�/�#�#�#�#�V�Q�V�^�b�i�o�w�z�{�|�|�~Ǉ�{�t�o�b�V�V�л̻Ȼϻлܻ�����ܻлллллллк�ݺֺɺºĺֺ̺��������	�������}�t�j�t�y¦²·µ²¯¦�������y�v�y�y���������������������������#���!�#�/�<�<�G�<�<�/�#�#�#�#�#�#�#�#�U�T�U�a�a�n�p�n�n�a�U�U�U�U�U�U�U�U�U�U����������������������������������������������������������$�1�3�8�:�>�=�0�����z�y�u�u�zÇÓÒÍÇ�z�z�z�z�z�z�z�z�z�z�����������������������������������������H�E�;�7�/�4�;�H�T�]�a�m�o�u�|�z�m�a�T�H�����y�m�R�K�T�y���Ŀݿ�������ѿĿ���|��������������������������������� ��
����"�&�)�1�)������4�����'�9�M�\�r���������������r�f�Y�4��ʾ����������s�{�������ʾ׾�������������Ʒ��������������������������̿�����������������������������4�3�)�4�9�A�G�M�O�M�C�A�4�4�4�4�4�4�4�4�����g�.�+�5�Z�g�s�������"�/�1�)��������������r�n�m�o��������������������������	���׾ҾҾ׾�	���"�.�8�;�?�5�.��	���������)�5�>�B�5�4�+�)�����_�Y�S�I�S�]�_�l�x���������z�x�l�_�_�_�_��	������(�5�B�Z�g�g�]�L�A�5�(���H�B�?�E�H�U�Y�a�c�g�a�U�H�H�H�H�H�H�H�H��úùõìàÚÝàìù�����������������Ž��������������������ĽȽĽ��������������Ľ����������Ľннѽݽ����ݽܽнĽ��6�.�*������*�6�C�N�G�C�6�6�6�6�6�6����ŹųŮŭŦŦūŭŹ��������������������	�������������	��-�/�;�@�H�H�;�/�'����������������Ľн����������н��������������(�4�<�?�4�3�(� �����������������������������������������������������������������
��%�*�%�#��
���񼘼r�f�`�r�y�~��������!�0�/�����ּ��������������������������ĿпѿտѿĿ������H�G�C�C�H�U�Z�[�Z�U�H�H�H�H�H�H�H�H�H�H����������	��"�.�;�@�>�;�0�"��	����������������	��������	��ŔŏŅłňŗŠŭŹ������������������ŹŔ�	���оƾþʾ����G�T�`�j�r�r�T�;��	�#�����#�0�4�3�0�#�#�#�#�#�#�#�#�#�#����������������������
����
�������f�_�Z�V�M�L�M�Z�f�s�������������s�f�f���������������ʾ׾����׾Ӿʾ��������r�g�r�u������������̼ҼμƼ���������r�L�B�>�7�4�6�@�L�Y�r�~�����������~�e�Y�L��������������(�2�5�;�A�C�A�5���.��$�:�y�������ýɽ׽�����н����l�G�.�����������������������������������������������~�v������������������������������Ľ����������ĽнӽؽҽнĽĽĽĽĽĽĽĺɺ������������ɺ�������	�����ֺɺ~�t�r�e�b�e�r�~�����������������~�~�~�~�b�U�I�=�<�5�<�H�U�b�l�n�{�łņŁ�{�n�bŇ�{�}ŇŇŔŠŭŰŵŭŭŠŔŇŇŇŇŇŇ�������������������������������������������%�'�-�4�@�@�@�4�0�'������ɺú����������ɺԺֺ�����ֺɺɺɺ�¿¦¦²¿�����������������¿�s�n�g�e�e�g�s���������z�s�s�s�s�s�s�s�s�����l�h�g�m�x��������������������������Ç�}ÂÇÍÓàìùù��������ùìàÓÇÇ������������������������������������ܻػѻܻ߻������'�+�1�'����������������ûϻû��������������������������������������������������������������������������������ùܹ�����ҹù������������������ĿȿȿĿ������������������6�)����������)�6�C�O�T�O�S�B�6����������������������� �������+�&�'�-�:�S�_�l���������������l�S�F�:�+�л����������������ûȻʻλܻ����ܻ�D�D�D�D�D�EEEEEED�D�D�D�D�D�D�D�D�E*EE)E*E7E=ECEPE\EiEoEmEiE\EPECE7E5E*E* ? z + " 7   , W * 7 3 U 2 ? ( M F = n ` 2 � 5 F < <  O m + ' � : ? 8 / E A f U p P N # Z 7 < 9 ` H 0 ] c 9 8 5 ( { 3 3 H E 2 6 4 L ; H ^ c = 8 0 S 3 D ^ O m    �  *  c  �  z  �  �    Y  �  a  k  �  �  i  �  r  r  �  �  @  P    U  �  �  o  �  ,  �  �  �  y  �  u  U  -  �  D  �  �  
  �  �    #  z  �  =    Y  �  �  �  �  s  \  �  �  �  j  u  �  �  X  �  X  �  �      K  o  9  �  a  �  �  <�1<�o<D����w��1�49X�o��o;o�o�o�#�
���
�y�#��t����㽣�
��P�ě��u��t��ixս8Q�0 ż�j��h�49X���C����t���`B�t��<j�q����w�C��C���C��t��C���7L�<j�y�#��%�o��w��w�,1��t��aG��P�`���T�0 Ž�%�#�
���
�'m�h�@�����8Q�ixս��-�T����vɽ�+�u���-�ixս�+�ě���O߽�1�������-���w����B��B0B@�B�B�IBڝB��BWB.C6B�BL�B$�B0�B*h�B�BK�B!�B ՐA�|B �NB=�A���B&<lB�9B#PBJ;B�B !B�B'��B)b9BH�B�RB!V�B"�zB&#B;�BppB,��B�(BB_B0�B	7�B|!B�KB
ZB��BjB=�B2B!� B�%B�aB�aB��B�!B�YB$xB�B�B��B)"�B#�lB
]>B˭B@�B�DBb�B�.B�B�uB6�B�Bd>B� B'5�B%8�Bq�B��B�B��B@3B��B��B�)Bi�B@�B.AXB�BA�B$;�BAeB* eB8�BC�B � B!OA�gB |�BHwA��B&{ B�B8�B>uB ^B&BC(B'��B)��B8�B6�B!>8B"B&?BƱBPB.>�B� B�B/�B	�]B�B��B9B��B=�B?�BʈB!�jB�B@�B�0BB(B��BH�B��B>5B�B(\B)>#B$B
?�BęB�zB�+B��B�MB��B�'B�B�7ByB��B'?�B% �B@$B�fA¦B�@���@G�5A�?Aq]:A�]@AƔ�A!�!B��A�/@��hA��WAxE�@�>A�;@�5AMۚB�A��yA:XiA��A�Z�A[.�A�T�@�IxA�֩A�EpA�*�A"�A(��A��wA�*�A�U�A'�SA4b^A�G[A�?@�_Av\�A�{A\�"AYv,A��AY�(A�YA��AB2�AO8�@���?��A��A�HA!ORA�NA'�g@B�o@��AA�ݨA�JH@�S}@84A�ŃA���@�7^A���B<+@�k�@��A��]>�Av�A�cAAҢ�@��&@�M�C�MkC��3A��B?�@�P�@J�A�~�Aq dA�zAƀ(A!�WB	<xA��.@��A���Az�9@�UDAԃ @�MAN�B3�A�	'A:��A��FA�~�A\&A��t@�ٞA�%�A�z�Ă�A!#�A)�B @A��6A�q A*=A5iA���A�x�A�Av��A�K�A[�vAZ��A��AWAꙙA�h�AB�AP�X@���?���A�l�A҅A!A��A'$W@C��@	�A�n�A�A��i@�C�@3�A��;A�� @�	�Aˋ�B�@�w@@�#FA��->>�Av�PA׀TA�zQ@��V@�>C�HC���            7            	      $            7      
   G               ,                         
               %            -         '      !   #               *         1            +         	                   ,      	         
   &                                                   '            9         5   )            ;   -                                    %            9         "      !   -                        A      #                           %                                                                           !            /         +   #            +   )                                    %            9                  !                        A                                 %                                             N�A#Nv��NM�;N���O�
N�N�S[M�@�NH�{O�$hN@�*N(dCOI��P#�N/��N�YP�&O��N�׃N�+N$��O�3�P#�OM�XN�h�N���OK
NF��N�DIN�I?N��N^�HO'�N���O���O��N���OC�2PL�"N���NLO;��O��O~�O�;�NFLO(�PN��N��DOz�O��OӇP��N��WOgmN[JsO���NK�N���N͍?N�NI�YN��O�v,N:jO�:�O?N���O��M�M�hYO��3NY/pO�*�N�~�O��@N�,�Nl!�N��j  *  "  .  	6  �  �  x  �  '  O  �  �  �    3    �  /    -  T  �  �    �  h  {  |  l  �  E       �    �  +  �    �  �  5  2    S  1  (  �  ,  �  �  �  g  �  �  q  $  m  �  �  �  p    z  �  �  {  &  �  k  �  �  �  �  �    e  	7  �=o<�9X<����`B;D��;ě�;�o;�o;�o��o�D���ě���`B��C��t��#�
��j��o�T���D���D����/��C���9X��t���t���/��1��9X��9X��j��j�ě������ͼ�j�ě��ě��ě����ͼ�/�0 ŽC���w��P��h��h��h��h���C��C��C��+�49X��P�0 Ž��8Q��w�H�9�'49X�<j�@��P�`�T���T���e`B�]/�ixսq���q���u��o�u��o��vɽ�v���������������������$)+59BIHB5.)'"$$$$$$knz�����ztnjkkkkkkkk)),*)#�������	�����.56?BNOUZYQNB55.....")6BDFBB6)(""""""""=BCOOQOBA7==========�����������������������
���������

���������������

������������&)5BEFINRZNB50{��������������{vor{�����������}��������������������������������������������������������������������RTUadgkmprrmaVTQPQRR�����������|}��������������������9AMTamqmmrtxvpaTL;99��	#0IUnwyuj<0'
���������������)))5BKJHB5*)���������������������������������������#/;<?<//# ��������������������GINSUbdkllfb\UNIGGGGw{�����������}{vwwww����������������������� 
##%#"
���������������������������� 
���������#0<IMTTSNI<10'#!�����znea[^anz������;BKNV[dpmhf_[NB@966;���������#("���������������������������inz����znciiiiiiiiii*6CJQTPOC6*MN[gtz����ytgf[NGGMM������������������������������������������������������������������

��������$$#�))/6<<;6.)!��� -10-)�������������������������� !%'%!��  ���#6O]aS6��������ntx��������{thnnnnnn��������������������/5;BGNSQNB53////////���������������~{z|���������������������������������������������������������������������������������{{{���������{x{{{{{����

��������Z[gt������������th\Z����� �������������".;BO[gorph[OB6)+5BHNUVSNNDB5.-,,)++��������������������������������������������������������������������������������9<HUanz����{znaUHB=9xzz��������zxxxxxxxx������������������������""����<IUcjswwrnbIHHF>;;:<##$09<800#��


���������������


 ������#�����#�'�/�<�?�H�R�H�>�<�/�#�#�#�#�V�S�V�_�b�k�o�{�|ǅ�{�r�o�b�V�V�V�V�V�V�л̻Ȼϻлܻ�����ܻлллллллк�޺ֺͺκֺ׺�����������������t�o�t�~¦®²µ´²­¦�������y�v�y�y���������������������������#���!�#�/�<�<�G�<�<�/�#�#�#�#�#�#�#�#�U�T�U�a�a�n�p�n�n�a�U�U�U�U�U�U�U�U�U�U�����������������������������������������������������������	��$�(�0�3�;�7�0��z�y�u�u�zÇÓÒÍÇ�z�z�z�z�z�z�z�z�z�z�����������������������������������������H�E�;�7�/�4�;�H�T�]�a�m�o�u�|�z�m�a�T�H�������w�b�l�y�����ѿݿ�������ѿ�����}��������������������������������� ��
����"�&�)�1�)������@����2�M�f�r���������������r�f�Y�@�����������{���������ʾ׾����׾ʾ�������������������������������������ٿ�����������������������������4�3�)�4�9�A�G�M�O�M�C�A�4�4�4�4�4�4�4�4��������q�u�������������	��$��	���������������v�r�q�t��������������������������	������ݾپ���	���"�.�4�8�.�"����������)�5�>�B�5�4�+�)�����_�Y�S�I�S�]�_�l�x���������z�x�l�_�_�_�_�(���������(�5�A�C�M�N�S�N�A�5�(�H�C�A�H�N�U�V�a�a�d�a�U�H�H�H�H�H�H�H�HùøìãàÛÞàìù��������ùùùùùù���������������������ĽȽĽ��������������Ľ����������Ľнݽ����ݽٽнĽĽĽ��6�.�*������*�6�C�N�G�C�6�6�6�6�6�6ŹŸŰŭŧŧŬŭŹ��������������������Ź�	�����������	���"�#�/�0�1�/�"��	�	���������������Ľ������ ������н��������������(�4�<�?�4�3�(� �����������������������������������������������������������������
��%�*�%�#��
���񼘼r�f�`�r�y�~��������!�0�/�����ּ��������������������������ĿпѿտѿĿ������H�G�C�C�H�U�Z�[�Z�U�H�H�H�H�H�H�H�H�H�H�	�����������	��"�#�-�1�2�/�.�'�"��	�����������	��������	�����ŠŔŌőŠŠŭŹ������������������ŹŭŠ����ھϾɾʾ׾����	��#�7�@�3�.��	���#�����#�0�4�3�0�#�#�#�#�#�#�#�#�#�#����������������������
����
�������f�_�Z�V�M�L�M�Z�f�s�������������s�f�f���������������ʾ׾����׾Ӿʾ��������r�g�r�u������������̼ҼμƼ���������r�L�F�A�:�8�=�L�Y�r�~�������������~�e�Y�L��������������(�0�5�9�A�5�(���.��.�;�y���������ĽȽֽ���н����l�G�.���������������������������������������������������������������������������������Ľ����������ĽнӽؽҽнĽĽĽĽĽĽĽĺ����������ɺ�������������ֺɺ��~�t�r�e�b�e�r�~�����������������~�~�~�~�n�f�b�U�I�G�A�I�U�W�b�n�s�{�}��{�v�n�nŇ�{�}ŇŇŔŠŭŰŵŭŭŠŔŇŇŇŇŇŇ���������������������������������������Ѽ�����%�'�-�4�@�@�@�4�0�'������ɺú����������ɺԺֺ�����ֺɺɺɺ�¿¦¦²¿�����������������¿�s�n�g�e�e�g�s���������z�s�s�s�s�s�s�s�s�������x�l�h�h�n�x����������������������Ç�}ÂÇÍÓàìùù��������ùìàÓÇÇ���������������������������������������߻������'�)�0�'�$�����������������ûϻû��������������������������������������������������������������������������������ùܹ�����ҹù������������������ĿȿȿĿ������������������6�)����������)�6�C�O�T�O�S�B�6�������������������������������+�&�'�-�:�S�_�l���������������l�S�F�:�+���������������ûлܻܻ����ܻлû���D�D�D�D�D�EEEEEED�D�D�D�D�D�D�D�D�E*EE)E*E7E=ECEPE\EiEoEmEiE\EPECE7E5E*E* ? g +  1   , W * 4 3 U 2 0 * M @ + l ` 2 t 1 G < <  . U + ' � ( = 9 / E A f U p 8 J  5 7 < 9 ` H ( V d 9  5 # { , 3 " E 2 6 4 F ; H = c = 8 0 S - D / O m    �  �  c  �  R  �  �    Y  �  a  k  �  �  P  �  �  �  7  �  @  �  �  �  �  �  �  `  �  �  �  �  ,  �  >  U  -  �  D  �  �  �  I  �  ~  #  z  �  =      @  �  �  �  s    �  �  �    u  �  �  X  l  X  �  (      K  o  9  �  a    �    =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  *  #        �  �  �  �  �  �  s  B  �  �  W     �  L   �  �          	  �  �  �  �  �    {  x  j  P  6    �  �  .  )  $      
  �  �  �  �  �  �  `  ?    �  �  �  j  1  �  2  �  �  �  	
  	#  	1  	6  	,  	  �  �  -  �  Q  �  ^  �  �  �  �  �  �  �  �  �  �  f  :    �  �  Y    �  /  g    �  �  �  ~  v  l  [  E  +    �  �  �  q  @    �  �  =  �  �  x  q  j  c  [  R  J  ?  3  '      �  �  �  �  �  y  B    �  �  �  �  `  ;    �  �  �  m  @    �  �  �  N     �   �  '  "               �   �   �   �   �   �   �   �   �   �   �   �  �  !  ;  L  M  @  )    �  �  �  �  �  C  �  b  �  d    �  �  �  �                     �  �  �  �  �  �  �  �  �  }  w  q  k  d  [  Q  H  ?  6  .  %            �  �  �  �  �  x  `  E  %  �  �  �  p  T  L  g  {  �  �  �  �  �  �  �  �      �  �  �  n  U  K  D  2    �    �  �  +  �  ,  0  3  .  '        �  �  �  �  �  �  �  �  �  �  y  _    �  �  �  �  {  _  A  "    �  �  �  [  %  �  �  �  X    j  �  �  �  �  �  �  �  A  �  �  T  J  .      �  �    (      &  .  .  +  "      �  �  �  �  �  e  J  (  �  p   �              �  �  �  �  �  c  -  �  �  r  '  �  `   �  -  "        �  �  �  �  �  �  �  �  �  r  e  X  J  =  0  T  N  H  B  +    �  �  �  �  �  �  q  X  ;      �  �  �  >  H  ?  5  2  `  �    U  %  �  �  Q  �  N  �  Q  �  �   �  �  �  �  �  �  �  �  m  Y  =    �  �  m  <    �  �  g    �  �  �      �  �  �  �  �  �  �  ]  4    �  =  �    {  �  �  �  �  �  �  z  i  V  C  /        �  �  �  �  �  �  h  I  ,    �  �  �  �  �  �  �    j  P  4    �  �  �  s  &  7  F  V  e  s  z  x  l  M  $  �  �  �  k  0  �  e  �  9  ,  L  g  u  |  |  u  m  b  V  J  =  1  "    �  �  �  �  w  E  R  a  i  ]  F  ,    �  �  �  �  e  5  �  �  ;  �  o    �  �  �  �  �  �  �  �  �  �  w  p  i  `  O  G  [  k  U  ?  A  D  A  8  (    �  �  �  �  {  Y  6    �  �  �  �  ~  x      �  �  �  �  �       �  �  �  �  �  �  y  Y  9     �  �  �  �  �  �  �  �  �  f  I  +    �  �  �  x  M    �  Y  �  "  K  a  n  y  �  �  |  o  Z  9    �  �  $  �  V  �   �         �  �  �  �  �  �  }  [  :  	  �  �  p  z  ?  �  ,  �  �  �  �  �  �  �  p  W  =  !    �  �  �  c    �  G   �  +      �  �  �  �  �  �  �  �  �  �  s  d  S  ?  *  "    �  �  �  �  q  Z  @        �  �  �  ~  _  2  �  �  �  �      r  h  P  2    �  �  �  h  c  P  !  �  �  �  d  �  �   X  �  �  �  �  �  �  �  �  g  I  )    �  �  �  "  �  �  D   �  �  �  �  �  �  �  �  �  �  �  �  ~  w  p  h  _  W  A  &    �  �  �      (  1  5  /  !    �  �  �  l  %  �  :  �  >  �  �    &  /  1  0  '      �  �  �  }  B    �  z  4  �  p  �  �  �  �  �      �  �  �  �  �  b  $  �  p    �  �  �  �  �  <  S  N  5       �  �  �  �  �  U  �  u  �  w   �  1  -  )  %                   
    �  �  �  �  �  �  (    �  �  �  �  �  r  T  6    �  �  �  �  �  �    �  �  �  �  t  d  T  B  2  #    �  �  �  �  �  ^  (  �  �  Y    ,  !      �  �  �  �  �  �  �  �  o  M  ,    �  �  G  �  �  �  �  �  ~  `  8    �  o    �  M  �  g  �  �      �  �  �  �  �  �  �  �  �  m  P  .    �  �  e  $  �  t  0  �  x  �  x  c  N  r  t  a  K  7       �  �  �  �  }  a  '  �  c  [  ^  L  <  !    �  �  �  j  -  �  i    �  �  �  �  $  �  �  �  �  �  �  �  x  h  X  B  '    �  �  Z  !  �  �  �  D  C  =  v  �  �  �  s  U  5    �  �  �  b    �  _  �  Z  q  g  ^  T  K  A  8  .  $         �   �   �   �   �   �   �   �  �    !        �  �  �  �    P     �  �  ?  �  x     `  m  _  Q  C  5  '                       �  �  �  �  |  �  �  �  �  �  �  �  �  {  S  '  �  �  �  E  �  e  �  )  �  �  �  �  �  y  _  E  .      �  �  �  �  g  6    �  �  �  �  �  �  �  �  n  �  �  �  �  �  z  V  *  �  �  V  �  1  p  _  N  =  -      �  �  �  �  �  U  *   �   �   �   �   �   �    �  �  �  �  �  �  �  �  �  �  m  N  +    �  �  �  m  C  z  k  O  '  �  �  �  O    [  T  F  '  �  �  b    �  ?  �  �  �  �  �  {  k  [  K  9  %    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  z  [  >  
  �  P  �  }    �  ;  �  �  �  {  p  d  V  G  9  +      �  �  �  Q    �  j    �  _    &      �  �  �  �  �    o  g  g  f  c  `  \  X  T  P  L  <  a  �  {  u  r  p  i  Y  8  	  �  �  u  B    �  �  >  �  k  _  R  F  :  -  !        �  �  �  �  �  �  �  �  �  �  �  �  r  Q  1    �  �  �  �  v  I    �  �  �  U  !  �  �  �  �  �  t  S  %  �  �  ~  =  �  �  c    �  g    �  '  �  �  �  �  �  q  W  9    �  �  {  <  �  �  �  J    �  �  e  �  �  �  w  [  ;    �  �  �  �  7  /    �  �  Q  �  �  1  �  �  �  �  �  �  �  �  �  �  z  Z  6    �  �  X    �  O      	  �  �  �  �  �  d  ?    �  �  f  /  �  �  P   �   �  �  �  �  F  `  Y  P  B  3  !  
  �  �  �  �  U  (  �  �  �  	7  	/  	/  	  �  �  V    �  �  L  �  �  o    �  R  �  �     �  �  Y  -    �  �  y  I    �  �  g  #  �  �  h    �  �