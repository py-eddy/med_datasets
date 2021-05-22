CDF       
      obs    M   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?�9XbM�     4  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M�^^   max       P�ϱ     4  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ���
   max       <u     4      effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?k��Q�   max       @E�            !H   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��(�`    max       @v�=p��
       -P   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @2         max       @P`           �  9X   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @ɯ        max       @�D          4  9�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �+   max       <o     4  ;(   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A��^   max       B-��     4  <\   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�l�   max       B-�\     4  =�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >�M�   max       C���     4  >�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >Gw�   max       C��     4  ?�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          �     4  A,   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          ?     4  B`   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          %     4  C�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M�^^   max       P ^_     4  D�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��c�	   max       ?�T`�d��     4  E�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ���
   max       <u     4  G0   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?�=p��
   max       @E޸Q�       Hd   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��z�G�     max       @v�Q��       Tl   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @-         max       @R            �  `t   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @ɯ        max       @�@         4  a   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         =�   max         =�     4  bD   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�IQ���   max       ?�R�<64       cx   
                              �   +            %   a               1   	            %                              $                                    
                        
            .            	               K                     NĜBN��fNno�M�^^Np�O�N��nO@ rN�r�O��P��MP61�O�b�NC`�N�EP7/]P�ϱN���O��2N�K�O�sP5'N���N^�4O|�Od��O�m
N�<�O[�O�N�)�N"#�O-�O�� O�*AN���P$׮N���Oc�OU�N�[�OmxNS��N?�{NaP6��O5%�Oj��N�u$N��Oj��O_=�NCOM/�N�jUN�'7OGO�j�N�3�N�O|�`N�ٳO��GN�c'N�g�O �?ON��O/OBr�O�e�N���O3�wO-�~Nq�OA��O.$6N���<u<u<49X:�o%�  ���
��`B�t��#�
�#�
�#�
�#�
�49X�49X�D���T���T���T���T���e`B�e`B�e`B�e`B��o��o��o��C���C���t����㼛�㼣�
���
��1��1��1��9X��9X��j��j��j�ě��ě����ͼ��ͼ�/��h��h��h�������o�+�C��C��t��0 Ž0 Ž0 Ž0 Ž8Q�8Q�8Q�8Q�P�`�Y��]/�e`B�ixս�%�����7L��hs��hs���
���
����������������7<FIUbchbaWUJI<99777��
 #&#
��#mmyz�����{zwmlmmmmmm>BFN[agmtx{tg[NMDB>>-/;<DHNRPHH<5/,'&(--����������������������������������������������������������������#"���������p~�������������{ukip��������+/0)������� ���������������������������������FRiz�����������zaTHFaz��������������n\[a\alnnsuyxtnjea[Z\\\\agnz{���������znebZa����� �������������������������������������������������������������������������������������
!%'&!
 �����#/HU]a_UHA<1/+#������

����������������

�������w{���������������|xw5BN`grppm[NB5.++,),5dgty���������tigddddzz|~�������zzzzzzzzz��������������������15BN[gt����yg[NB;5,1����������������������������������������GIQamz��������wqaTLG56?BJO[cb[TOJB:65555TXamvz~������zma\VST��������������������#?Uamz�����zrna\UMH=?����������������������������������������#)-6BBBB6)##########3BRg���������t[D@303��������������������s|���������������tnstt��������tolltttttt
)696)







��������������������'0<Ubiopx{qnbUTI>30'��������������������OY[_ho{tythc[OJDEHO������������������������ ������������� 
#'05:30#
����dgt�����������zgb_ad��������������������'05<IJUUOII<800%''''16<CEIMU[_^\VI<30-.1 ���        +5BNQ[gt����tgN6)$��� 
"#&##
�����QU[anz�zynkaUONQQQQ�������
#)#!�����v{����������������zv���������������������������������!!#(1/*#!
�������������������������dgpt���������xtlg^_d������������������������������������������������������ #+,.3=ADFC<5/#/;<HUaefa[UH<81/,-+/�}¦²¸¿����¿¸²¦�ʼü����������ʼѼּ��������ּʼʽ��	���������!�#�.�1�.�+�!��ĳĩĦģĦĳĿ����ĿĳĳĳĳĳĳĳĳĳĳĚęčČĄčĚĦĪĳĳĳĮĦĚĚĚĚĚĚƎƊƁ�w�{ƁƇƎƕƚƧƱưƯƨƧƚƓƎƎ�����"�&�/�;�H�T�T�W�T�O�H�;�/�"�������������������
��(�/�4�1�/�,�#��
���-�:�F�G�S�W�S�O�F�:�-�!���!�-�-�-�-�-�S�Q�F�C�H�L�S�_�l�x�������������x�l�_�S�л�������4���������������r�@�'��п����g�`�^�d�{�������Ŀѿ������ݿĿ����$�������������0�=�I�W�Q�E�C�5�-�$ƳƭƯƳ����������������ƳƳƳƳƳƳƳƳ����������������������������������������������������������	���0�4�?�M�X�V�H������r�L�B�E�Y�~���ɺ��%����غ˺������N�F�A�A�A�N�Z�g�s�������s�p�g�Z�N�N�N�N�������������������������������������������������������(�)�(�&� �����ìàÕÎÇÂ�z�t�uÀÇÓÞìù������ùì���r�Y�M�/�@�M�Y�r�������ּ�����ʼ����������������'�.�'����������������������"��������������ſżųŴż������������������������)�%�&�&�"�)�6�B�L�\�h�l�h�f�[�S�O�A�6�)��پľ��������ʾ׾޾�����������������������������������������������������Ľ������Ľɽн۽ݽ�����������ݽн������������������6�J�L�G�6�*�����������������|�z���������������������������������������������������FcF`FBF=F1F$FFF$F1F=FJFVF_FeFmFoFwFoFc����߿����������$�&�!����꿸��������������������ĿԿ߿�����ݿ�����������*�6�C�F�C�6�4�8�*�'���Z�5������ݿؿݿ���5�R�d�s�������s�Z�a�`�U�R�U�V�_�a�n�u�z��z�v�n�d�a�a�a�a����ƴƪƧƣƧƳ��������� �� �����������h�f�P�L�O�V�\�b�h�uƁƎƓƚơƚƎƁ�u�h�O�D�B�7�<�B�O�[�[�h�i�h�[�T�O�O�O�O�O�O���������������������
���
� �����������������#�&�%�'�#��������������������������������������������������������������������������������������������������������)�+���(�0��������������������
��!�#�0�2�<�0�(�#��
�����������������������	����#�����	���/�/�+�/�<�H�L�U�V�\�[�U�H�<�/�/�/�/�/�/��߹�����������������������b�V�_�n�|ŔŠŹ����������ŹŭŠŇ�{�n�b�-�+�#�&�*�-�:�F�S�_�l�x�v�y�x�e�F�:�1�-�����������������������������������������������������ùϹܹ�� �	������ܹϹù��s�m�g�Y�W�Z�g�s�����������������������s�����������������������������������������лƻû������û̻лܻ�������ܻл�ĳĮĦĠĦĮĿ��������������������Ŀĳ���������������������������������������ؽ������������������ĽĽнĽ������������������ݽս����(�4�A�M�]�b�Z�S�A�(��׾Ҿ־׾����������׾׾׾׾׾׾׾׾��ݾӾҾ׾������	�����	�������0�/�'�$���$�'�0�=�=�G�I�J�I�E�=�=�0�0�������������������������ɾȾ������������������������������������������������������������������Ŀѿݿ����ݿѿƿĿ�������������������$�"�$�-�0�4�0�$��ùöìãèìù������������������������ùD�D�D�D�D�EE E7EPEVE\E`E\EXENE(EED�D��<�/�2�<�H�U�^�a�e�a�U�H�<�<�<�<�<�<�<�<àÜÓÑÏÇ�z�w�zÇÓìû��������ùìà�����������ļмּ������������ּʼ����������Ǻɺֺ��ֺֺɺ��������������������������!�-�M�S�_�h�a�S�F�-�!�E�E�E�E�E�E�E�E�FFF$F1F:F1F-FE�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E� ` I b Z L I : 7 + < e 3 i g B ? < \ + O ^ Q R 2 J E . l J ? V S e I 8 I ] C C ; 8 K � r ` e 0 $ & � q P M 2 ` S M $ M - [ 5 l M R u I 9 ' b K [ z \ j w c       �  �    |  ;  *  �  �  h     D  {  y  �  B  �  �  �  ?  8  �  �  m    �  �  o  {  �    U  �  E  �  =  \  �  �  �  �  �  �  �  O  �  �  �  �  �  T    y  �  9    `  X  �  �  W  �  �      �  �  *  �    �  �  �  �    �  N;ě�<o;ě���o�ě��T�����
�C���C���j�+�Y��\)�e`B�u�H�9��G�����+��t����}󶼴9X��1�C���P�Y��ě��\)�@��ě���/�49X�+�H�9���ͽm�h�����\)��h�,1����`B��`B�e`B�D���8Q���+�D���0 Ž��m�h�49X��w�<j��7L�]/�]/��E��T���u�ixսY���%���
��C���hs�+�������罥�T���-��Q���`��9XB��B&̓B$p~B&WA���B�^B�UB��B-ZB6B�B*d�B��B�B!	A�P?B�uB�9B��B# B��B _%B 2B�B�B�B�B�B)�B��B
>�B��B��B�(B��B:3A��^B�NA���BESBۅB��B FJB��B�mB	��B��B�B�;B��BuBB'7BB 1�BvB��B��B$�<B
T�B�<B&NXB&��Bx�B�*B��B�B#�9B��B�B��Bu�B
�)B
5�B-��B!��B"[JB-�B:xB��B&��B$?�B3A���BːB��BJB,ԓB�B�-B*9qB�]B�;B!1�A��\B�#B�]B��BU�B�UB 5�B @�BϣB�B�aBVB�B)�}B��B
5DB�3B�4B��BɛBA��3B@A�l�Bz�B��B��B ��B�VB�B	yaB�OB?�B��B��B@�B&��B AqBBB�SB�B$�B
?B��B&@B&��B=�B	��B4�B0�B#��B�[B�B/�B@B
�BB
1rB-�\B!?�B"J�B?�B��A�#A&AAᗩA���B�A�)�A��@z�%@��@�J�AuvfB
@BAJ,yA��L@.�5A�HA�)�A�\�A�_�@��\@��{A�$oA���A�,�AS�A�U�A*�A� �A�c�?-��C���A��Awq�A���A�ΓA��zB�BB��Aٔ�A��*A���At(Avc�A���A趎A�a�A�@'?	uA�+ @�(�A��>�M�A�
�A���@�MA��A���A"�@A7��AV��AYsB
\�AL4yA�p�Ayt�B	�A�FvC�z�AĤA�A�W@4)h@r�PC���C�&�A��A ��A�A�e8Aߎ�B4lA�xTA��&@{
�@��"@��Au3+B	��BgAJ��A���@;��A���A���A��Á3@��@�ZA��A�xlA��UAT�<A���A+_A��GA�|�?/ynC��A���Ay �A�dxA��(AƀB�iBA�A�r�A���A���At4Av�A���A�RA���AĄ�?K#A�@|�A  �>Gw�A�vA�]E@��A�A�A!�dA77�AVfAZ��B
?�AL�7A��Ay�B�A�c�C�X8A�s�Á�A`�@-@@k̝C���C��   
               	               �   ,            %   b               1   	            %                              %                                    
                                    /            	               L                     	                                 ?   3   '         1   =               3               #                        #      -                           7                                                                        #         !                                             %   %   #            %               %                                       #                                 #                                                                                             NĜBN���Nno�M�^^Np�N��BN���O@ rN�r�O��O�|�O�
O�;NC`�N#��O�> P ^_N���OT�N�K�Oq]�O���N���N^�4O=f�O�OqN�<�O[�O��N�)�N"#�OR�O�� O�*AN���O��N`��Oc�O2�N�[�OmxN%N?�{NaO�O͞Oj��N�u$N��Nհ#O8`NCOs�N�p�N�'7OGO��NL�LN�
O81�N�ٳOs{N���N�g�N��OC��N���OBr�O��N���OJ`O��Nq�O0�BO.$6N���  �  R  �  �  �  `  4  p  m  3  0  �  �  �    W  :  �  �  {         �  5  �  J  �  �  9  U  �  �  \  �  #  �  �  �  �  '  �  @     �  �  "  y  �  ]  �  �    �  V  �  q  !    �  L  �  �    �  �  i  \  �  �  �  x  O  �  }  -  �<u<e`B<49X:�o%�  ��`B�#�
�t��#�
�#�
������9X�T���49X�T����h�L�ͼT����t��e`B��o��j�e`B��o���㼴9X��/��C���t����ͼ��㼣�
��j��1��1��1���ě���j���ͼ�j�ě����ͼ��ͼ��ͽC�����h��h����P�o�o��w�\)�C��t��<j�8Q�49X�P�`�8Q�<j�<j�8Q�Y��]/�q���e`B��+��%��C���C���hs��t����
���
����������������8<IUa^USIB<;:8888888��
 #&#
��#mmyz�����{zwmlmmmmmmNN[]gjtuvtg[RNKFNNNN-/<=HJOMH<0/**------��������������������������������������������������������������������������v���������������vtv�������)--)�������� ���������������������������������SV]amyz�������zma[TS��������������������\alnnsuyxtnjea[Z\\\\`cinz|����������zni`����� ����������������
��������������������������������������������������������������������
#%%$
����!#/<HUWUTLHC<1/%#!������
�����������������

�������w{���������������|xw5BN[chffeb]NB52//0.5dgty���������tigddddzz|~�������zzzzzzzzz��������������������15BN[gt����yg[NB;5,1����������������������������������������SX_amz������zma[TRRSABEOZ[`[POMB<7AAAAAATXamvz~������zma\VST��������������������#?Uamz�����zrna\UMH=?����������������������������������������#)-6BBBB6)##########<BN[gt���������g[N9<��������������������s|���������������tnstt��������tolltttttt
)696)







��������������������1<IUbhmnrtnlbUIE?<51��������������������LOU[htuutmpth[YROLIL������������������������ ������������� 
#'05:30#
����fgt�����������ukgdbf��������������������)06<GISSMIE<910())))0025<IUXZ\[YURI<:200 ���        %0BNX[gt����tg[N;5*%��
 #%#"
��������QU[anz�zynkaUONQQQQ�������
"
�����x}����������������zx�����������������������������������
"')#
������������������������ggst����������tgdbgg����������������������������������������������������� #+,.3=ADFC<5/#/;<HUaefa[UH<81/,-+/�}¦²¸¿����¿¸²¦�ʼż����ʼּ�������ּʼʼʼʼʼʽ��	���������!�#�.�1�.�+�!��ĳĩĦģĦĳĿ����ĿĳĳĳĳĳĳĳĳĳĳĚęčČĄčĚĦĪĳĳĳĮĦĚĚĚĚĚĚƁƀ�ƁƊƎƙƚƧƪƬƬƧƟƚƎƁƁƁƁ�"�� �"�,�/�;�H�N�S�H�G�;�/�"�"�"�"�"�"�����������������
��(�/�4�1�/�,�#��
���-�:�F�G�S�W�S�O�F�:�-�!���!�-�-�-�-�-�S�Q�F�C�H�L�S�_�l�x�������������x�l�_�S�4�������'�4�M�f����������r�f�M�4�������r�m�p�y�������Ŀ׿����ݿĿ����$�����������0�=�I�S�O�D�B�<�4�0�$ƳƭƯƳ����������������ƳƳƳƳƳƳƳƳ�����������������������������������������"��	��������������	��"�/�<�C�A�;�/�"�������w�x���������ɺֺ��������⺽���N�F�A�A�A�N�Z�g�s�������s�p�g�Z�N�N�N�N�������������������������������������������������������(�)�(�&� �����à×ÐÇ�z�w�zÄÇÓÚìùþ������ùìà�r�_�S�I�S�f�r������ռ޼�Ѽʼ��������r�������������'�.�'����������������������"��������������ŹŸŸ����������������������������)�)�)�)�,�0�6�>�B�O�S�[�e�[�Y�O�M�B�6�)�׾ʾ��������¾ʾ׾�������������������������������������������������������Ľ������Ľɽн۽ݽ�����������ݽн�������������������*�C�F�?�6�*���������������|�z���������������������������������������������������FJFEF=F1F$FF!F$F/F1F=FJFVF]FcFcFkFcFVFJ����߿����������$�&�!����꿸��������������������ĿԿ߿�����ݿ�����������*�6�C�F�C�6�4�8�*�'���5�(����
�	��(�5�A�N�V�d�l�g�W�N�A�5�U�T�U�X�a�d�n�s�z�|�z�t�n�a�U�U�U�U�U�U����ƴƪƧƣƧƳ��������� �� �����������u�o�h�\�U�Q�X�\�e�h�u�|ƁƊƎƏƐƎ�|�u�O�D�B�7�<�B�O�[�[�h�i�h�[�T�O�O�O�O�O�O���������������������
���
� ����������������#�&�#�"�#�%�#�����������������������������������������������������������������������������������������������������������������&�%������������������
���#�-�0�2�0�$�#��
�����������������������	����#�����	���/�/�+�/�<�H�L�U�V�\�[�U�H�<�/�/�/�/�/�/��߹����������������������Ňł�{�{�{ņŇŔŠŭŹ��ŹŸŭŠřŔŇŇ�-�%�&�'�*�-�:�G�S�_�g�l�r�u�_�S�F�A�:�-�����������������������������������������ù��������¹ùϹйܹ�����������ܹϹ��s�o�g�Z�Z�Y�Z�g�s�������������������s�s�����������������������������������������лƻû������û̻лܻ�������ܻл�ĳıĩģĦķĿ������������ ����������ĳ���������������������������������������ؽ��������������������Ľ̽Ľ��������������4�(����������(�4�A�H�M�X�\�Z�M�A�4�׾Ҿ־׾����������׾׾׾׾׾׾׾׾��׾׾�������	������	������0�(�$���$�*�0�;�=�F�I�D�=�0�0�0�0�0�0�������������������������ɾȾ��������������������������������������������������𿸿������������Ŀѿݿ����ݿѿſĿ��������������
�����$�&�$�"�����ùöìãèìù������������������������ùD�D�D�D�D�D�D�EEEE7ECEPEVETEJE8EED��<�/�2�<�H�U�^�a�e�a�U�H�<�<�<�<�<�<�<�<àßÓÓÒÌÑÓàéìùùÿ��üùìàà�ʼ��ƼʼѼּ���������������ּʺ��������Ǻɺֺ��ֺֺɺ����������������!���������-�:�K�S�_�f�_�Y�S�F�:�-�!E�E�E�E�E�E�E�E�FFF$F1F:F1F-FE�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E� ` : b Z L H ; 7 + < = * d g A 9 6 \ 6 O \ F R 2 J @ # l J J V S P I 8 I F A C  8 K � r ` b . $ & � V E M > U S M , 7 ' G 5 p D R h H = ' _ K B j \ p w c       �  �    |  �  �  �  �  h  4      y  D    9  �  �  ?  %    �  m  �  K  �  o  {  m    U  @  E  �  =    v  �  J  �  �  �  �  O  i  R  �  �  �    �  y  #  �    `  �  n  �  �  �  %       >  �  �  �  �  �  8  �  �  �  �  N  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  �  �  �  �  �  �  �  �  l  T  :      �  �  �  �  h  F  $  9  B  K  Q  N  K  D  :  0       �  �  �  �  �  t  `  O  ?  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  {  q  f  \  Q  F  <  1  &            �  �  �  �  �  �  �  n  Z  I  8  &          �  �  �  �  �  d  @    �  R  W  ]  _  _  _  Y  T  G  8  #    �  �  �  �  d  Q  R  S  �    $  .  2  4  2  *      �  �  �  �  f  #  �  u     �  p  k  `  O  7    �  �  y  5  �  �  =  �  �  0  �  {       m  d  [  R  J  C  7  *      �  �  �  �  �  �  �  z  s  m  3  /  +  &      #      �  �  �  �  �  �  �  �    a  >  
�  @  �  1  o  �  �    /  �  �  @  �  K  
�  	�  �  �  �  �  E  s  �  �  �  �  �  �  �  o  D    �  �  L  �  �    �   �  �  �  �  �  �  �  �  w  S  '  �  �  �  W  +  �  �  #  �    �  �  �  �  �  �  �  �  �  �  �  �  |  r  f  Z  M  A  5  )   �                              �   �   �   �   �   �  ?  l  �  �  �    ;  N  V  T  6    �  �  �  Y    �  �    ,  U  Z  d  �    #  8  4      !    �  �  �    A  Z  C  �  �  �  |  t  l  d  ]  V  P  I  B  ;  2  %    
  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  }  u  [  n  �  p  A    {  v  q  l  f  b  b  a  a  `  Z  N  B  6  )     �   �   �   �  �  �  �  �  �  �  �  l  @  +    �  �  {  ]  =  �  p  �   �  Y  �  �        �  �  {  (  �  h  (  a  `  9  �  �  f  o    	    �  �  �  �  �  �  �  i  F  #    �  �  �  �  Q  !  �  �  �  �  �  �  �  �  �  �  }  v  n  f  ^  V  a  p  �  �  0  #    .  ,       �  �  �  �  k  H  #  �  �  �  �  s  D  �  �  �  �  �  �  �  �  �  �  �  Z  .    �  �  �  K    �  _  �  �  "  :  G  H  I  >  &    �  �  �  \    �  @  �  -  �  �  �  �  �  �  �  �  �  �  �  �  h  I    �  �  P   �   �  �  �  �  �  �  |  r  e  V  E  2      �  �  �  ^    �  m  �    )  5  9  6  .  "      �  �  |  #  �  ^  �  �  S  �  U  O  J  E  @  =  :  7  6  7  8  9  3  '      �  �  �  a  �  �  �  �  �  �  �  �  �  m  V  >  %    �  �  �  �  �  d  �  �  �  �  �  �  e  .  �  �  �  c  #  �  �  I    �  �  M  \  Q  D  4  #      �  �  �  �  �  �  �  �  �  x  D   �   �  �  �  �  �  d  F  +    �  �  �  o  2  �  �  �  G  �      #        �  �  �  �  �  �  �  �  �  �  z  g  R  <  '    Q  b  x  �  ~  w  r  |  �  �  �  m  E    �  l    �  �   �  i  �  �  �  �  �  x  e  R  <  $  	  �  �  �  x    �  ?  �  �  �  �  �  �  �  �  �  {  f  M  -  	  �  �  �  V    �  >  �  �  �  �  �  �  �  �  �  �  �  �  s  ]  F  -    �  �  �  '        �  �  �  �  �  �  �  �  u  e  U  E  5  &    	  �  �  �  �  �  �  �  �  �  z  h  R  6    �  �  _  �  �   �  3  7  ;  @  H  R  [  k    �  �  �  �      '     �  `        �  �  �  �  �  �  �  �  �  �  �  z  h  U  B  .       �  �  �  �  �  �  �  �  �  �  �  z  m  `  T  K  A  8  /  %    �  �  �  �  �  �  �  �  �  �  �  �  �  z  T  ,  �  �  �   �      "      �  �  �  �  �  }  N    �  �  M  1    �  �  y  q  `  L  8  #    �  �  �  �  �  �  �  |  k  P  $  �  �  �  �  �  �  �  �  �  �  �  �  �  �    }  y  v  s  q  l  g  ]  M  =  ,      �  �  �  �  �  �  �  ~  i  T  ?  )     �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  e  >    �  �  @  �  �  �  �  �  �  �  �  �  y  i  [  n  w  g  N  1    �  �          �  �  �  �  �  �  m  M  ,  
  �  �  �  v  M  %  M  {  �  �  �  �  �  �  k  8    z  q  \  @    �  �  �  V  N  R  U  H  ?  C  E  B  9  &    �  �  �  �  �  �  q  Q  /  �  u  h  Z  K  :  )      �  �  �  �  �  �  �  �    }  z  q  o  m  i  d  \  R  F  8  &    �  �  �  �  {  U  .  �  �  �    !         �  �  �  �  z  c  L  1    �  t  �  m   �  �  �          �  �  �  �  �  m  H    �  �  �  G     �  �  �  �  �  �  �  �  �  �  �  �  o  P  -    �  �  �    \  }  �    @  F  <  ,    �  �  �  J  �  �  C  �  w    i  �  �  y  j  [  J  :  (      �  �  �  �  �  q  M  *    �  �  �  �  �  �  �  �  �  �  �  �  �  �  z  V  (  �  �  v  )  �        �  �  �  �  �  ~  e  H  '    �  �  �  O    �  �  �  �  �  �  �  �  l  V  Y  `  ]  Q  D  /    �  �  �  n  (  �  �  �  �  �  �  �  �  �  �  w  c  K  2    �  �      �  d  ]  :  4  .  %    �  �  �  y  B    �  X  �  ~    d  �  �    '  4  G  L  S  \  Z  <    �  �  3  �    t  �  �   �  �  �  �  �  �  �  �  u  d  P  <  $  
  �  �  �  v  -  �  [  Q  �  �  �  �    >  �  |  I  �    
B  	�  	�  �  S  �  A  �  �  �  �  |  |  ~    �  �  �    u  c  L  .  �  �  o    �    [  n  w  w  j  Q  .    �  �  x  C    �  |    �  �  N  �  8  =        �  �  z  Z  �  �  c  6    �  �  S  
  �    �  �  �  �  �  �  �  �  �  �  x  i  Z  K  =  .    	  �  �  z  }  a  <      �  �  �  �  �  �  �  �  �  �  �  �  �  �  -    �  �  �  Z    �  �  I  V  i  a  v    �    �  �  [  �  �  �  �  �  �  �  �  i  Q  5    �  �  �  }  Q    �  