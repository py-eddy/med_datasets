CDF       
      obs    L   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?�XbM��     0  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M���   max       PLF�     0  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       �ě�   max       <���     0      effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?&ffffg   max       @F\(��     �  !<   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��33334    max       @vm\(�     �  -   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @.         max       @Q�           �  8�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @Α        max       @��@         0  9�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �n�   max       <�C�     0  :�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�i�   max       B4��     0  ;�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A��#   max       B4�P     0  =$   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >���   max       C�8�     0  >T   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >E6#   max       C�8T     0  ?�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          c     0  @�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          1     0  A�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          #     0  C   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M���   max       P �o     0  DD   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��<64   max       ?Ծ�(��     0  Et   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ���   max       <���     0  F�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?&ffffg   max       @F�G�z�     �  G�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       �ə����    max       @vl��
=p     �  S�   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @'         max       @Q�           �  _�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @Ͱ        max       @�}@         0  `,   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         C�   max         C�     0  a\   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�hr� Ĝ   max       ?Լj~��#     �  b�   	         
   	            	                  ,         2                              ,            _   '                  B      &            c         A            )   	   A                                                
      %      C   0   N�,WN���NfO�N{��OD�O��;N֜�O��NL��N&#N���N��/N8�M���PI��On}N��UP��N�W�N*�N�H�N
7�N5�/Ok�"N�(�O$*!N�ObP�O?UO�P�Ok�
PLF�P	p�O~��O�~�N;NB�O��JP�nN+�gP;�N͜O�hyN�@=PDCOׇNT��O�l�O��O	,N}V�O�<�N7+�PpYO"�O�Ne�IOh�ZN3P�Of=Ok��O��]Nb��N�f�O�N�w�N��3NI�9Nͅ�N��N�rO?��N��O�R�O���Ny+�<���<�t�<e`B;�o:�o%   �o�D����o��`B��`B�o�t��D���T���e`B�u�u�u��C����㼛�㼣�
���
��9X��9X�ě��ě��ě����ͼ��ͼ��ͼ��ͼ���������/��`B��`B��`B�����+�+�C��\)��w��w�#�
�#�
�,1�,1�0 Ž0 Ž0 Ž0 Ž8Q�<j�@��L�ͽP�`�P�`�P�`�Y��aG��e`B�ixսm�h�q���u��o�����C����P�����ě��ě�)5BGNSRNB5)##%/03563/&#IN[cgmhg[NMIIIIIIIIIUH<:89;<HHOUVUUUUUUUqz��������������~zpq#<HP`cieaUH=/! ""�����������������������������������������������������������������������067BLOWWTOGB:6000000��������������������������������������������������Wet�������������pZSW�����������������������


����������6B[lz�����th[O<0-+-6����������������������������������������)6BIORQOBA61)!+/7;AHIHHC;/++++++++enxz~���|znleeeeeeee����������������������������������������TUanz�����zrna\YUTTuz{���������zvttuuuu:B[t��������tg`NIB=:������

��������amz����������zma[Y[a��������������������HYz������znaUH<A@>?H )6BOht����~thOB)" ����������������������������������������&),6>BFB96.)&&&&&&&&xz������zwxxxxxxxxxxz����������������zvz.6FW[ht�����h[JJB7*.��������������������5BNgt���������tg[7.5����������������������������������������������������������������,35/���������9BHNT[^gtytqjgb[NB:9FIU_bdb^bcb^UIIDFFFF����������������������#07<ACKE9+
�����y{�����������{yxvvyy������������������������

������#/<??</%#3<anz���������whU<23��������������������-03<DIPUbhhb_UI<0(%-������������������������������������������������������������ )5BNTX[ZNB5)%Z[gt��������tgc[ZYZZ��������������������tt��������}utstttttt������������������������������������������������������ 
��������������������������������������������������15@BNQQNMBB5+-111111{������������v{{{{{{z�����������������~z
#).)$#"

$)/7<FMOMIC</#����������������������������������V�S�O�O�S�V�_�b�o�r�x�y�s�o�c�b�V�V�V�VŹŵŷŹ���������������������������Ź�������������ĿſǿſĿ�����������������������������ĿĺĿĿ������������������������������������������������������������������������������������������������������������������������
�
�������������������������������������������U�Q�H�F�H�I�U�^�a�l�k�a�U�U�U�U�U�U�U�U�Ϲ͹ȹϹܹ����ܹϹϹϹϹϹϹϹϹϹ��a�W�U�K�U�W�a�n�z�}�z�w�n�h�a�a�a�a�a�a�*�$�*�5�6�B�C�O�\�h�s�u�~�u�h�\�O�C�6�*������������� ��������������������s�o�s�����������������s�s�s�s�s�s�s�s��׾˾ľо׾־�	��"�:�A�S�Z�T�.�����4�(��������(�;�A�M�Z�Z�S�M�I�M�4�t�m�l�s�t�z�}�t�t�t�t�t�t�J�G�W�m�y�������Ŀѿ׿ؿԿ˿������m�`�J�H�<�3�/�#�"�#�/�<�H�U�[�a�e�d�a�[�X�U�Hùóìéìíù����������ùùùùùùùù�������#�/�1�;�<�>�<�8�/�#���������������������������������������������H�B�<�9�<�@�H�T�U�U�U�U�H�H�H�H�H�H�H�H�.�"��	����"�.�;�G�T�_�`�Y�T�J�;�3�.�U�M�U�V�b�l�n�{ŇŋŎŇ�{�z�n�b�U�U�U�U������������������������
���
��������Ç�{�z�x�zÅÇÓØàìòóìàÓÇÇÇÇ�g�_�[�]�s���������������������������s�g����������������������������������������ƿƬƜƐƎƚƧ�������������	������ƿ���������������$�0�=�A�@�6�0�$�"���G�N�Q�W�s���������������������������g�G�s�h�g�k�l�g�b�e�s���������������������s��޿ѿʿ����������ѿ�������������4� �(�1�;�A�N�Z�g�s�����������w�g�N�A�4��������������������������������������������������� ��
�����������������������������������������
��#�.�,�(�(��������ù��������Ϲܹ��'�.�1�6�3�'�������O�K�K�B�?�B�O�[�_�e�[�R�O�O�O�O�O�O�O�O�H�8�4�7�=�J�T�m�������������������z�a�H�h�^�[�O�F�O�U�Y�[�a�h�t�wāĊĂā�t�h�h�Ŀ��������������Ŀѿݿ���������ݿѿĺ~�}�r�e�_�^�e�r�~���������������~�~�~�~���������ּ����!�/�1�/�+�!����㼽������������ŹůŭŦŪŭųŹ���������������һ-�*�*�-�2�:�C�F�P�S�T�S�P�H�F�:�-�-�-�-��������$�'�@�M�\�f�q�~���v�Y�M�'����ܻƻɻл������'�@�R�Q�D�<�4�����������������ûлܻ����ܻлû�����������'�4�@�F�@�:�4�'��������Ŀ��������Ŀѿݿ�����������ݿѿ�������(�/�3�(�$�����������������������ù��'�<�@�E�E�A����ѹ����L�E�A�A�F�L�Y�e�m�r�{�z�r�g�e�Y�L�L�L�L�����������������������������������������T�P�O�R�T�X�a�e�e�e�h�a�T�T�T�T�T�T�T�TŔŉņŔřţŭŹ����������������ŹŭŠŔ�
���
��#�#�(�#��
�
�
�
�
�
�
�
�
�
�z�v�n�f�b�a�n�zÇÓàëðìäàÓÏÇ�z�������������������
�����
�����������<�0�#����#�0�<�I�U�g�n�s�n�l�b�U�I�<ƧƦƚƘƚƧƳ����������ƺƳƧƧƧƧƧƧ�=�;�0�/�0�;�=�I�O�V�b�f�k�b�V�I�=�=�=�=�	�������������	��"�.�/�2�.�-�"��	�������������!�'�*�!������������Ľ����������������ĽȽͽн׽ݽ�ݽнĽĺ����������!�"�!�������������������������������$�0�1�0�(�%�$������������������)�*�/�)�������ìèàÞÚàìù����������ùìììììì�������������������������ʾ˾Ͼξɾ�����E7E4E4E7E=ECEPE\EiEqEiE`E\EPEOECE7E7E7E7E�E�E�EuEsEuE�E�E�E�E�E�E�E�E�E�E�E�E�E��f�Y�M�H�F�G�K�M�Y�f�r�������������x�r�f���������������������������������������� P ] : ? V N N < B A A f o � ' ? Q A P 3 C K a + \ . J W T f   ( E [ = 6 W 7 + Z \ K . A G g = 1 3 ( V 7 I v ! N � 1 9  , R X D 9 < d K _ V / C I 8 ! 6  .  P  �  �  e  �    =  Z  >  �  X  O  �  W    �  q    I    D  m  �  �  d  �  �  �  �  �  f  |  K  l  :  9  E  �  i  �  �    �  !  T  {  �  4  /  �  )  k  �  #  \  �  �  =  �  �  #  �  �  B  �    Y    �  �  �  �  3  6  ~<�C�;D��<49X���
�o��o�u�#�
�D���T�����ͼT���49X�u�ixռ��ͼ�1��o��h��j������1�����#�
��h����󶽋C��49X�0 ŽaG�����%�0 ŽT�������,1����\)��C��D���T���49X�1'�@��0 Ž�����hs�D���8Q콩��T����/��%�T���T����C��u���-��7L��+�u�y�#���P���㽍O߽�%��t������1�����Ƨ�hs�n����B6�B�B��B)@BZ�B�BIWB:�B��BBP�B�Bo�BJ�BB�BV�B"B!C�B�~B A�i�B'�B!�B*yB��B �B	ͮB��A��xBy�B�B�B��B�eB��B 7QB_B��B�B	X�B�OB*\�B!B-��B��B')cB �|B$��B)j�B)�BC�B/YB\B��B&�[B֧B�B��ByUB	�8B��B
MB��BjB�B#�B"�B>3B�CB
��B4��BVsB7B�B�VB9�B�eB�"B:<B��B8�B@B ��B�sB:�BA0B?"B@`B@{B
@kB<wB;B<�B!:�B��BL�A��#BF�B!��B?�B�0B 5�B	ƑB;�A�y�B�NB�[B�B�2B@8B�8A���B ��B��B1�B��B��B*D�B ��B-�tB��B'(CB �7B$~�B)G�B)��B?;BAB�B=�B&��BBIB8�B�
B?�B	��B=[B
@B�B|�B�NB"��B"�BNB�B
��B4�PB@B9�B&B�B3�A��~Av��A�@�A� A��SA�K�A��>A��>�ָA��BV�A�؞AF��AZL�A8�A�/�Aq��Aī�A͂+A���A���Aē�Aa�?A�w�A�6Aʩ�A�͡A�B�,B	v
A��A�IA|��A�~�A�h�A��%A���?"D`Aٟ�A�?AۉA{�Z@�A�A��Q@zHg@�	4@�v|@��@ɰrA}<�A���>���?�rMA�
_A�[A�SLA��FAɞhA�.=A��BH�B�A\XA�:A'J]@]��B�A�A�!�AL�qC��SC�8�@ߒ;@��BB�A��<Av��A䔠AЈ�A���A��A�QAŁ�>���A�oBx�A���AH�AZ �A7aFA��LAq�A��A�{A�\�A�k#AĆpAa�SA�h�A���A���A�}�A�}�BK�B	��A�n�A�X�A}�jA�u�A�� A�{%A��.?/CA�w�A��!AۃKA{Z@�?AMA�t7@{"�@Ψ�@���@�'�@�L�A{AA�x�>E6#?��A�ȺA�|aA�K�A�#A�uA�}�A��gB�cB}oA\��A
qA&��@[�%B��A��A�w�AL�ZC��%C�8T@��S@���   
         
   
            	                  -         2                              -            `   '                  C      &            c   	      A             )   
   B                                                
      %      D   0                                                1   !      %                              '            -   )      !            )      '            +         #   '               /                                                                                    !                           #         #                              !            #   #                  #      !            !                           #                                                                  N�,WN���NfO�N{��O ��O�(N�$O��NL��N&#N.O2N��/N8�M���P �oOIa�N��UO���NY!N*�N�H�N
7�N5�/Ok�"N�(�O$*!N�ObO�6N�P�O�P�O(��O��.O� oOhw�O�6�N;NB�O��JO�*�N+�gO�?SN͜O�ްN�@=O�S�OׇN�OND�O��NO	,N}V�Of_N7+�OcO"�O�Ne�IOh�ZN3P�Of=OUOW��Nb��Nz��O�N���N��3NI�9NvnN��N�rO8qiN^WO�GOdH]Ny+�  �  �  �  -  �  #  6     M    �  U  0  �  �  e    k  G  �  �  n  �  �  '  /  �  �  �  �  T  	�  �  �  �  �  J  �  ?     �  `  p    
�  D  m  	.  �  �  �  �    	`  ]  1  K  !  �  �     �  >  �  �  �    M  �  �  B  �  �  �  N  �<���<u<e`B;�o%   �o�ě��D����o��`B�49X�o�t��D���ě��u�u�������㼋C����㼛�㼣�
���
��9X��9X�ě�����h���ͼ��T����h��/��`B��/��`B��`B�49X���+�+�C��C��u��w�#�
�}�@��,1�,1�<j�0 Že`B�0 Ž8Q�<j�@��L�ͽP�`�T���Y��Y��e`B�e`B�q���m�h�q�������o�����O߽����ě����ě�)5BGNSRNB5)##,/1441/#IN[cgmhg[NMIIIIIIIIIUH<:89;<HHOUVUUUUUUUsz����������������zs #/<FOU^`gaUHA/#"$% ������������������������������������������������������������������������46<BHOTSOB?644444444��������������������������������������������������gt������������{td]\g�����������������������


����������36BO[cpx}��{th[OB;33����������������������������������������)6BIORQOBA61)!+/7;AHIHHC;/++++++++enxz~���|znleeeeeeee����������������������������������������TUanz�����zrna\YUTTuz{���������zvttuuuuGN[t����������te[SJG�������

��������amz����������zma[Y[a��������������������GLUanz��������naUMGG#)6O[t����{th[OB)##����������������������������������������&),6>BFB96.)&&&&&&&&xz������zwxxxxxxxxxxz����������������zvz6:AFO[ht{�����vh[RB6��������������������5BNgt��������tg[;5/5�����������������������������������������������������������������#'%�������9BHNT[^gtytqjgb[NB:9GIU\bcbXUQIEGGGGGGGG�����������������������
#0;><.*#
����y{�����������{yxvvyy������������������������

������#/<??</%#8<Hanz�����ztnUH>858��������������������-03<DIPUbhhb_UI<0(%-������������������������������������������������������������ )5BNTX[ZNB5)%[[gtx�������tgd\[ZZ[��������������������tt��������}utstttttt������������������������������������������������������ 
��������������������������������������������������15@BNQQNMBB5+-111111{������������v{{{{{{{�����������������{
#&+$#
#/;<BCA<:/#����������������������������������V�S�O�O�S�V�_�b�o�r�x�y�s�o�c�b�V�V�V�V��ż����������������������������������ƿ������������ĿſǿſĿ�����������������������������ĿĺĿĿ�����������������������������������������������������������������������������������������������������������������������������������������������������������������������U�Q�H�F�H�I�U�^�a�l�k�a�U�U�U�U�U�U�U�U�Ϲ͹ȹϹܹ����ܹϹϹϹϹϹϹϹϹϹ��a�^�U�O�U�[�a�n�x�r�n�c�a�a�a�a�a�a�a�a�*�$�*�5�6�B�C�O�\�h�s�u�~�u�h�\�O�C�6�*������������� ��������������������s�o�s�����������������s�s�s�s�s�s�s�s��پҾѾ۾����	�"�1�<�@�6�.�(��	����4�.�(���������(�4�A�M�V�L�E�A�4�t�m�l�s�t�z�}�t�t�t�t�t�t�m�`�[�]�c�m�y���������Ŀ̿ҿοƿ������m�H�@�<�7�8�<�H�U�_�X�U�U�H�H�H�H�H�H�H�Hùóìéìíù����������ùùùùùùùù�������#�/�1�;�<�>�<�8�/�#���������������������������������������������H�B�<�9�<�@�H�T�U�U�U�U�H�H�H�H�H�H�H�H�.�"��	����"�.�;�G�T�_�`�Y�T�J�;�3�.�U�M�U�V�b�l�n�{ŇŋŎŇ�{�z�n�b�U�U�U�U������������������������
���
��������Ç�{�z�x�zÅÇÓØàìòóìàÓÇÇÇÇ�s�h�c�f�m�y���������������������������s����������������������������������������ƿƬƜƐƎƚƧ�������������	������ƿ��������������$�0�7�=�=�<�2�0�$���s�g�^�[�Y�Z�a�s�����������������������s�s�l�j�r�p�g�l�s�����������������������s��ݿѿ̿����������Ŀѿ��� ���������'�/�4�=�A�N�Z�g�s�����������u�g�N�A�5�'��������������������������������������������������� ��
�����������������������������������������
��#�.�,�(�(���������ܹϹ��������ùϹܹ����'�(�(�����O�K�K�B�?�B�O�[�_�e�[�R�O�O�O�O�O�O�O�O�K�9�5�8�?�K�T�a�m���������������z�u�a�K�h�^�[�O�F�O�U�Y�[�a�h�t�wāĊĂā�t�h�h�Ŀ��������������Ŀѿݿ��������ݿѿĺ~�}�r�e�_�^�e�r�~���������������~�~�~�~�ʼ������ƼѼ߼�����$�'�%� ���������������ŹůŭŦŪŭųŹ���������������һ-�+�+�-�7�:�=�F�I�F�C�:�-�-�-�-�-�-�-�-������'�0�4�@�M�Y�f�o�q�f�Y�M�4�'��������������'�4�@�L�I�9�4�'����������������ûлܻ����ܻлû�����������'�4�@�F�@�:�4�'��������Ŀ����������Ŀʿѿݿ���������ݿѿ�������(�/�3�(�$�������������������������ùܺ��������޹Ϲù����L�E�A�A�F�L�Y�e�m�r�{�z�r�g�e�Y�L�L�L�L�����������������������������������������T�P�O�R�T�X�a�e�e�e�h�a�T�T�T�T�T�T�T�TŔŉņŔřţŭŹ����������������ŹŭŠŔ�
���
��#�#�(�#��
�
�
�
�
�
�
�
�
�
�z�v�n�f�b�a�n�zÇÓàëðìäàÓÏÇ�z���������������������
����
�����������:�0�"�����#�0�<�I�U�d�n�n�i�b�U�I�:ƧƦƚƘƚƧƳ����������ƺƳƧƧƧƧƧƧ�I�>�=�7�=�C�I�L�V�b�d�i�b�V�I�I�I�I�I�I�	�������������	��"�.�/�2�.�-�"��	���������!�%�(�!�����������Ľ����������������ĽȽͽн׽ݽ�ݽнĽĺ����������!�"�!�������������������������������������������������������������)�*�/�)�������ìèàÞÚàìù����������ùìììììì�������������������������˾ξ;ʾȾ�����ECE8E7E6E7E@ECEPE\E^E]E\EPEIECECECECECECE�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E��Y�Q�M�J�H�I�M�N�Y�f�r����������z�r�f�Y���������������������������������������� P V : ? S O E < B A F f o � ' < Q > B 3 C K a + \ . J J D f   @ \ 8 6 W 7 & Z Q K - A : g 7 ? & ( V * I F ! N � 1 9  ' T X = 9 ; d K H V / B A ( , 6  .  �  �  �  4  p  ;  =  Z  >  S  X  O  �  =  �  �  �  }  I    D  m  �  �  d  �      �  b  �  /    6  :  9  E  �  i  1  �    �  �  T  0  �  1  /  �  �  k  �  #  \  �  �  =  �  �  �  �  }  B  �    Y  ?  �  �  �  s  1  �  ~  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  �  �  �  �  �  �  �  �  �  �  �  {  i  Q  /    �  �  3   �  �  �  �  �  �  �  �  �  �  n  I    �  �  c    �  J  �  r  �  �  �  �  �  �  �  �  �  �  �  �  �  |  g  R  =  (     �  -      �  �  �  a  0    �  �  �  �  b  :    �  �  H  �  �  �  �  �  �  �  �  �  h  M  4    �  �  �  �  �  �  x  P      "        �  �  �  �  �  �  h  D  %  �  �  �  N   �  �  �  �  �       9  L  U  O  &  �  �  {  ;  �  �  l  "   �                   
    �  �  �  �  �            M  <  ,    �  �  �  �  �  `  F  6  $    �  �  �  z  T  .        �  �  �  �  �  �  �  p  W  >  #    �  �  �  �  _  s  �  �  �  �  �  �  �  �  o  G    �  n    �  O  �  {  
  U  R  O  L  H  ?  6  ,  "      �  �  �  �  �  �  �  �  �  0  0  0  0  0  0  1  1  1  1  /  *  %  !          	    �  �  �  �  �  s  e  \  U  N  G  @  9  /          �   �   �    ?  i  �  �  �  �  �  x  e  I  *    �  �  �  R  �  -  �  T  ^  d  _  [  W  M  C  8  ,         �  �  �  x  J  7  .       �  �  �  �  �  �  m  S  =  +      �  �  �  �  �  �  �    E  ^  j  h  `  F    �  �  R    �  8  �    M  �   �  �      !  A  F  G  E  <  /    �  �  �  A  �  �    �    �  �  �  �    j  V  <      �  �  �  W     �  �  u  :     �  �  �  �  �  �    _  ?       �  �  �  �  �  �  �  �  �  n  g  _  W  P  H  A  9  2  *       �   �   �   �   �   �   �   �  �  �  �  �  |  b  H  -    �  �  �  �  t  P  ,    �  �  u  �  �  �  r  d  P  C  7  +      �  �  �  �    T  #  �  �  '  !      	  �  �  �  �  �  �  t  K  #  �  �  �  �  �    /  #    �  �  �  �  }  g  J  (    �  �  �  �  �  h  G    �  �  �  �    h  W  K  >  +      �  �  �  �  �  C  �  �  �  �  �  �  �  �  �  �  j  Q  2    �  �  �  @  �  |  �    o  �  �  �  �  �  �  �  �  ~  a  ?    �  u    �    �  0  �  �  �  �  �  �  �  �  �  �  �  l  6  �  �  ]     �     �  �  3  I  R  S  M  @  .    �  �  �  \    �  k    �  	  M  m  	Y  	�  	�  	�  	�  	�  	�  	�  	�  	B  �  |  	  �    q  z  D  �  �  �  �  �  �  �  �  p  T  6    �  �  b  	  �  +  �  �  �  �  �  �  �  �  �  �  �  �  �  ~  Y  2    �  �  �  O  �  N  �  �  �  �  �  �  �  �  }  Z  3    �  �  i  :  �  �    m  �  �  �  �  �  �  �  �  �  �  �  |  m  Y  ;     �   �   �   �  J  H  G  F  D  C  B  A  ?  >  7  +        �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  k  D    �  �  �  W    �  .  �  �  !  9  ?  "  �  �  �  y  Y  1  �  �    �  �     T     �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  s  L    �  z  !  �  p    �  ~  _    7  `  R  B  -       �  �  �  v  �  ]  1    �  �  �  }  _  @  o  p  o  i  Z  D  )    �  �  �  �  �  X  .    �  �  �  �    �  �  �  �  �  i  N  4      �  �  �  �  �  �  �  �  �  	�  
C  
s  
�  
�  
�  
�  
�  
s  
@  	�  	�  	5  �    P  �  �  @  �  D  >  8  -      !  ,       �  �  �  �  b    �  \   �   �  D  M  U  ^  f  k  d  ]  V  O  F  :  /  $    �  �  �  _  .  U  �  �  �  	  	#  	-  	,  	   	  �  �  -  �  *  �  �  c  ,    �  �  �  �  �  �  �  �  �  �  �  �  x  z  g  F    �  �  �  �  �  �  �  �  �  �  �  �  |  f  N  5     �   �   �   �   t   U  �  �  �  �  �  �  �  �  }  t  k  b  Y  N  ?  0  !       �  K  �  �  �  �  m  P  +    �  �  k  !  �  g  �  z  �  H  �    �  �  �  �  �  �  �  �  v  i  [  K  8       �  �  �  �  	*  	0  �  �  	_  	C  	7  	  �  �  h    �  .  �  	  Y  f  b  �  ]  S  @  #    �  �  �  �  d  3  �  �  �  F  �  |    �    1  ,  &           �  �  �  �  �  �  �  u  \  D     �   �  K  G  C  ?  G  R  \  e  m  t  a  3    �  �  �  �  �  h  O  !        �  �  �  �  �  {  K    �  �  Q  �  �  R  �  �  �  �  �  �  �  �  �  �  �  }  c  G  &    �  �  �  k  B    �  �  �  ~  j  T  ?  %    �  �  j  5  �  �  Z  �  3  z   �          �  �  �  �  z  [  2    �  �  )  �  �  :   �   �  �  �  �  �  �  �  �  �  �  q  Z  <    �  �  �  L    �  t  >  '    �  �  �  �  h  A    �  �  �  a  =  (     �   �   �  �  �  �  �  �  �  �  �  s  g  Z  L  >  0  "      �  �  �  �  �  �  �  g  G  '    �  �  �  t  L  #  �  �  �  f  B    �  �  �  �  �  �  �  �  �  t  Y  >  #    �  �  �  T    �    �  �  �  �  �  y  [  :    �  �  �  �  �  l  <  �  �  I  M  9  &       �  �  �  �  �  ~  t  k  a  X  O  G  ?  7  /  :  ;  =  @  V  o  �  �  �  �  �  �  �  �  �  �  �  p  �    �  t  g  Y  K  >  0  "      �  �  �  �  �  �  ^  ,  �  �  B  0    	  �  �  �  �  w  ]  4    �  �  g  ,  �  f  �  G  �  �  �  �  �  e  K  -    �  �  �  X    �  a  �  h  �  �  X  �  �  �  �  �  �  v  7  �  �  R  �  �  E  �  �  *  �  E  
�  �    N    �  �  �  l  7  �  �    
�  
  	�  �  <  u  y  C  D  K  M  A  .    �  �  �  Y  $  �  �  V  �  \  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  }  p  e  Y    �  F