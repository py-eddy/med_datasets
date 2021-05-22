CDF       
      obs    I   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?�ȴ9Xb     $  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       Nx   max       P���     $  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       �Ƨ�   max       =�P     $  �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?B�\(��   max       @F�z�G�     h  !   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�z�G�    max       @v�\(�     h  ,�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @)         max       @Q`           �  7�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�R        max       @��          $  8|   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ���m   max       <���     $  9�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�.�   max       B4xU     $  :�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�m�   max       B4��     $  ;�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       ?�{p   max       C���     $  =   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       =�n   max       C���     $  >0   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          Z     $  ?T   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          A     $  @x   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          =     $  A�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       Nx   max       P���     $  B�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?������   max       ?Ԟ쿱[X     $  C�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       �Ƨ�   max       =�P     $  E   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?Q��R   max       @F�z�G�     h  F,   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�Q���   max       @v�\(�     h  Q�   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @'         max       @Q`           �  \�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�R        max       @���         $  ]�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         D�   max         D�     $  ^�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�$tS��N   max       ?ԛ��S��     �  _�         	      3   S      	      +   A      -         Y   3      @   ,         
                     
         (      8         *            
      *                3               +   H                           
      	   (               
   OW�N��`N�mN�ZkP�KP���O�=N^ɉN�o�O��mP^��PFDOv?�N��O��ZP�{O��O���PA��Pe{�ND��N-�N���N-9�P,N���N��N�ygN�mN�B�O2NxO��O%��O��O�l�O�æO��ND'xNlCNw/�N��xN��O�l�O�PO@yOD��Oޯ�Pt�O?�}N���N�n�OB3�P��O�
�N��sO?�N��N�[N*)^O4��NAs�N���N��O�g�N�ϵO��OS2NL�
N�"�N��1NA�O�~=�P<���<t�;ě�;D��;o��o���
�ě���`B��`B�o�#�
�#�
�#�
�49X�D���T����o��o��o��o��o��o��C����㼛�㼼j���ͼ��ͼ���������`B��h���������o�t���w�#�
�#�
�,1�,1�0 Ž<j�@��H�9�H�9�H�9�H�9�L�ͽP�`�P�`�T���]/�]/�e`B�e`B�ixսm�h�u�u�u�y�#�}�}󶽃o���罬1��1�Ƨ���������������������-/<HUU\XUH<4/#------����������������������������������������*5BN[gif[B5�������
#<{����wnU<����#/<HRSNHGB<//$#�����������),)' ;HTasuzvvy��zaTH;67;������������������}����������������yv}&,5BFN[dgheb][NB5)%&@BOS[^ghihe[OB@=<=@@��������������������������)4;;)�������������������������������������FMapz��������zaTKDEFx����������������{tx"#/<A?<8/#! """""""""#/7<FG<//+#""""""""����������������������������������������BO[kt�����m[OB)�����������W[ahmtvvwtkh^[[OWWWW������������������������������������������������������������ )6CDLONKIC:6+*?CNOPWYOCB???????????HUnz����������qZUC?-/3<AH_aca[ZUH<1/*'-)5BN[gouxxtg[NB5).5Not���~ug[NB52,)).�������������������������������������������������������������������������������MOQ[_fhjih[POLMMMMMM~����������������}|~;<EHUajiea_UH=<9;;;;DN[hz�������tg[F<;<D������������������cipt������������thac�����
	�������������
#0<@=@F<#����5Ng�����������t[NB55��������������������������������������������������������������������������������^o�������������R_YQ^�����������������@BNO[`hltv{th_[QOB@@��������������������))468@BEFGDB=6))%'))����������������������������������������$/1<HPU_eigaUHB<6/*$zz����znlnozzzzzzzzz��������������������������	���������#0<IU_b[ZUI<0)��������������������������������	  �����,6BIOR[\[VOGB6,,,,,,�����������������������


�����������

);N[svtg[UPNB:51)$$)�����������������
������
����������ŭūŨŧŬŭŹ������������Źŭŭŭŭŭŭ�ݽѽ۽ݽ��� ������ݽݽݽݽݽݽݽݽݽݾ����������������������Ⱦɾ������������������s�a�R�O�W�s�����������������������������������^�T�J�T�s��������������	���)�%�$�#�&�)�6�B�M�O�[�a�\�[�O�N�B�6�)�)��������������� �������������������������������������������������������������ļĠĉčĚĦĳĿ�������
����������ļ�#�����"�8�U�nŔŝŖťŧŢŔ�{�b�I�#�y�m�O�H�G�`�m�������Ŀ˿ݿ��ۿѿ����y�b�V�J�?�=�7�8�=�I�V�b�o�{�ǈǉǈ�x�o�b�`�[�W�`�i�m�y�|�����������������y�m�`�`�Z�M�A�8�2�0�4�A�M�����������������s�Z�"���������5�X�a�m�������������^�H�;�"����������������$�0�8�A�D�B�8�0�$�����	����ʾ������ʾ����	����'�!��	�5�������߿����N�]�z���������s�A�5�h�C�:�-�&�&�-�,�.�6�E�\ƚƳ������Ƴƚ�h�M�M�K�L�M�Z�a�f�k�i�f�Z�M�M�M�M�M�M�M�M�
�	��
���#�)�#�#��
�
�
�
�
�
�
�
�
�������(�5�7�=�5�4�(�������àÞÙàáìñùüùòìàààààààà�����ʾӾӾʾ����ʾ����"�:�?�=��׾����a�a�U�M�J�O�S�U�a�h�n�r�u�z�{�z�n�b�a�a�H�B�;�2�;�C�H�T�a�c�m�t�m�a�`�T�H�H�H�H�)�%�&�����)�6�B�M�M�I�G�B�6�)�)�)�)�����������������	����	��������������x�u�l�k�f�i�l�x�y�����������x�x�x�x�x�x�	����������	�"�.�7�;�;�.�.�%�"��	�G�>�;�:�;�G�T�_�V�T�G�G�G�G�G�G�G�G�G�G�s�q�z�{�s�g�_�\�^�g�s�����������������s�����������������������������������������ݿο������������������ѿ�����������������+�%�*�4�C�O�L�K�O�N�F�C�6�*�������������������2�7�5�9�9�;�0�*��Y�P�K�@�=�:�L�Y�r�����������������~�r�Y���������ɺֺ�ֺܺɺ��������������������"��"�)�/�7�;�D�H�J�H�D�;�/�"�"�"�"�"�"�Y�O�L�H�L�Y�^�e�r�s�{�r�n�e�Y�Y�Y�Y�Y�YŠŚŔŇŃ�{�n�d�m�uŔŠŬŭŹżſŹŭŠ�����ݾ߾�����	���	��������׾ɾ̾׾�����	��"�*�)�&�!��	����׿	���������������	��-�5�7�3�.�"��	�����������������������������ƽͽͽ˽Ľ����������ּ�������������ʼ������Ľ�������������������"�����нȽ���������������)�5�I�N�S�Q�J�I�'������N�K�L�B�A�B�N�[�g�t�v�x�t�g�[�R�N�I�D�=�<�=�B�I�V�b�o�r�y�u�o�b�V�I�I�I�I�����������ʾ׾����׾ʾ���������������������������4�@�M�Y�]�Y�M�4��čā�j�g�vĉćĞ�����������
�<���ĿĦč�f�Y�F�4�'� ��'�4�M�f�������������r�f�������������������ûлػԻл̻û����������������¿Ŀѿݿ�����������ݿѿĿ����_�^�S�Q�S�_�c�l�x���������������x�l�_�_��������������"������������������6�6�+�6�B�O�P�O�G�B�6�6�6�6�6�6�6�6�6�6FF	FFF%F1F7FJFVFcFoFsFtFdFVFKF=F1F$F���������ĿϿϿȿĿ�������������������������������)�+�6�;�?�6�)����������������������ûлػԻлû������������������������ûܻ�����������ܻû��������������������������������/��
����.�G�S�`���������~�t�h�S�G�/��ݽĽ������Ľʽн����&�!�����������������������������������������������������~�r�j�p�r�~������������������������D�D�D�D�D�D�EEEEEEED�D�D�D�D�D�D�E*EE*E4E7ECEPETETEPECE7E*E*E*E*E*E*E*E*�G�<�3�2�<�H�I�U�e�n�zÍÒÇÃ�z�o�a�U�G Y Y I T V Z / d @ N 2 6 , 4 S J  M ] 2 5 r F ( f ? 6 = B p * 2 \ 6 > B ` $ < W 8 { - I . M z D A F  F m L \ l H E P P � 0 Q �  e < ^ y - P o I    l  �  4  �  �  �  P  �  �  ,  �  �  �        y  O  �  �  _  x  �  H    	  �    �  �  �  "    m  �  �  �  �  W  Y  s  d  �  �  )  �  k  =  �  �    �    W  �  �  a  �  �  4    [  �  "  �  %  R    �  �  �  z  ;<���;D��%   ��o�D�����T�ě��T���T���H�9��t���w�]/��C���w���ͽ}�\)�����u��1���
���ͼ�9X�0 Ž\)���ͽ\)�\)�\)�'�`B��C��,1�� ŽL�ͽP�`��t���w���Y��H�9�T����1���P����m�h��������q����o�]/���������m��7L��C���C��}󶽁%����������w��\)���
��O߽�����-��\)�ě���������FBx3B�(B)�mB4xUB�B%�EBy�B97Bu�A�.�B~�B*}�B� B�aB!�BѻBB�DA��CB�{B>FB^�B'�B��B�B�?B�PBBX�B!k�B0>�B1�B�
B��B��B�B<
B"+5B!�B�VB@CB
��B�1B	K�BȍB�5B-u�B%f�B
O�B��B�0B�B�fBzBL$BJB�vBאBY�B @B<BT�B.�BeB&,�B�.BmB	BB�B�_B-*B�B%�B�wB;kB)�IB4��B�2B&�(B�]B��BD�A�m�BA5B*A�B�DB��B!_B�NB9B��A�}uB��B6MB?�B?�B��B�VBGB�#B�WB?�B!@B0:7B1>�B@%B5�B�9B�nBq B"?�B!�<B/�B@ B7&B��B	<%B�lB��B-�JB%;KB
~�BE�B�AB�B�!B
>BA#B��B�$B�;B�2B</B�'BGB?�B?�B%�B5HB^`B��B�\B��B@B��B?�A��TA�lmA-	AM��A���A�!�A�ZA��At��A�}bA��-Aq0fB@XAmK�AC�kA��FB	ZAW�MA�EB�\A>�LA�Z{A���A�8.AV�2A�RA�z�A�`�A�;�@��*A[�@AeɈA�B�A���Ax#&A�_�A��?��J@3�:A��_?�{pA�AX;�AY�A[�FA#$�A/�A*��A��<A���B��AP��@�f/A�|p@�F�@�RA|>@���A�ćA�XOC���Ax#�Aյ@�0@�w�BS0A�:A/��A���@=C�N�C��NA�A��4A��A,��AM�A��A��A׃�A�rRAuHLA��A�9An�:BA�AmO�A@��A�}�B	J�AY��A�	B�OA?�A���A���A�(hAY3.A�~�A���A֐�A���@�AZc�Af��A���A���Av�MA��A���?��Y@3�/A���?�VA�{�AX=�AY�3A\�|A#<AU�A+�>A�v�A��[B/�AP��@��fA�yU@�t�@���Az�k@��hA�T�A�q�=�nAx|UAՀ�@� w@���B�^A"^A0�A���@ۣC�IC���A�/�         
      4   T      	      +   B       -         Z   3      @   -         
                              )      8         *            
      +   !            3               ,   I                                 	   (                                 )   A            #   /   -         #   =         -   1               5                        %      #      %   !                  !            %   +               A   #                              #      )                                 %   =               !   #            %         #   '               3                                                                     %                  =                                 #      '                  OW�N��`N�mN�ZkO���P���N��N^ɉN�o�O���O�_RO��;O,u�N��O��P̃OL3�O"��OΫvP$$�ND��N-�N���N-9�P 8�N:��N���N�2�N�mNp{O�6NxOv��N�5Ot��O/�FO��JOj�ND'xNlCNw/�N��xN���O��O}.#OikOD��O��uO�Y�O٬Nʃ�N�n�OB3�P��O��N��sO?�N�b{N�[N*)^O4��NAs�N�P�N��O�g�N�ϵO� N�h�NL�
N�"�N�>�NA�O�~  z  �  A  E      �    �  �  �  w  Z  �  _  �  i  �  u  ^    �  �  �  �    v  �  �  E     �  �  o  �  �  �  �  	  �  /  X  �  �    �  S  5  �  �  e  �  �  ~  �  R  �  q  �  7  �  �  G  1  d  �  �  �  �  y  0    a=�P<���<t�;ě����
�#�
���
���
�ě������`B��o����49X�u�<j��/��1��P���ͼ�o��o��o��o���
�ě����
�������ͼ�����/�����t����8Q�t��+�0 Žo�t���w�#�
�,1�<j�49X�<j�<j�H�9��O߽P�`�P�`�H�9�L�ͽT���e`B�T���]/�aG��e`B�e`B�ixսm�h��%�u�u�y�#��%��+��o���置{��1�Ƨ���������������������-/<HUU\XUH<4/#------����������������������������������������	&5BO[adb[NB5)��	#0In{���{bT<
���#/<HIPKHD?<4/(#�����������),)' HSTamnpruvmgaTHB=<<H������������������������������������~���05<BN[]a`^[WNB53))00?BEOQ[\ehb[OEB?>????�����������������������������������������������������������������RV^hz�������zmaTRPPR���������������{{}�"#/<A?<8/#! """""""""#/7<FG<//+#""""""""����������������������������������������)BO[hot����}[NB6,"���������������Y[chktuuvtjha_[TYYYY������������������������������������������������������������!**6CKNMKHDC6/*?CNOPWYOCB??????????aoz�������������zdaa,/<EHPTUWXUOH<5/-),,8?BEN[gnrsqmg[NIB;5845BLNP[aeca[NB850-.4�������������������������������������������������������������������������������MOQ[_fhjih[POLMMMMMM~����������������}|~;<=HLUaadaa]UH?<;;;;>BHN[ct������tg[J@>>��������� ����������ghkrt~�����������thg�����
	������������
#0;>><B<0#���X]gmt�����������te\X��������������������������������������������������������������������������������^p�������������T`YQ^�������������������@BNO[`hltv{th_[QOB@@��������������������))467@BDEFDB?6*)&'))����������������������������������������$/1<HPU_eigaUHB<6/*$zz����znlnozzzzzzzzz��������������������������	���������#0<IU_b[ZUI<0)����������������������������������
����,6BIOR[\[VOGB6,,,,,,�����������������������


�����������

);N[svtg[UPNB:51)$$)�����������������
������
����������ŭūŨŧŬŭŹ������������Źŭŭŭŭŭŭ�ݽѽ۽ݽ��� ������ݽݽݽݽݽݽݽݽݽݾ����������������������Ⱦɾ����������������������j�]�Z�e�s�����������������������������s�e�_�Y�Z�b�s����������� ���	���)�'�&�%�)�)�6�B�H�O�[�^�[�W�O�I�B�6�)�)��������������� �������������������������������������������������������������ĴĦĤħıĳ����������
������������Ĵ�I�<�)�'�)�/�<�I�U�b�n�{ņŇŌŎł�n�b�I�y�m�`�W�O�R�`�m�������ǿϿԿҿÿ������y�V�O�I�D�=�F�I�V�b�o�v�{ǃǃ��{�o�b�V�V�m�a�`�\�`�l�m�y���������������y�m�m�m�m�M�>�9�>�M�Z�s�����������������s�f�Z�M�8�/�!��� �;�H�V�a�m�z������z�s�T�H�8��������� ����$�)�0�9�>�=�:�0�$����������׾;ƾʾ׾����	������	���5�(���������5�A�N�a�s�y���y�s�g�A�5�K�;�4�6�C�O�\�uƚưƻ����ƸƳƧƚ�u�h�K�M�M�K�L�M�Z�a�f�k�i�f�Z�M�M�M�M�M�M�M�M�
�	��
���#�)�#�#��
�
�
�
�
�
�
�
�
�������(�5�7�=�5�4�(�������àÞÙàáìñùüùòìàààààààà�����ξ־׾Ҿʾ����ʾ���	�5�;�4���׾��U�R�O�U�a�n�o�n�n�a�U�U�U�U�U�U�U�U�U�U�H�E�;�6�;�F�H�T�a�b�m�r�m�a�[�T�H�H�H�H�)�)�)�,�)�$��)�6�B�B�F�E�D�B�6�)�)�)�)�����������������	����	��������������x�v�l�l�g�k�l�t�x�����������x�x�x�x�x�x��	����������	��"�.�0�7�.�,�"�"��G�>�;�:�;�G�T�_�V�T�G�G�G�G�G�G�G�G�G�G�����y�l�g�c�`�f�s�������������������������������������������������������������ſĿ������������������Ŀѿۿ����ݿѿ����
������*�6�C�D�I�I�C�@�6�*�����	���������������0�5�2�7�6�8�*��Y�M�H�J�Y�e�r�~�����������������~�r�e�Y���������ɺֺ�ֺܺɺ��������������������"��"�)�/�7�;�D�H�J�H�D�;�/�"�"�"�"�"�"�Y�O�L�H�L�Y�^�e�r�s�{�r�n�e�Y�Y�Y�Y�Y�YŠŚŔŇŃ�{�n�d�m�uŔŠŬŭŹżſŹŭŠ����������������	���	� �����������׾ξо׾����	��"�'�&�#���	����	�������������	��+�4�5�1�.�"���	�����������������������������ýĽʽʽĽ����������ּ�������������ʼ������Ľ������������������ ������߽н��������������������)�5�:�<�6�-����t�o�g�[�R�N�E�C�N�[�]�g�t�u�w�}�t�I�H�@�F�I�V�b�o�p�w�s�o�b�V�I�I�I�I�I�I�����������ʾ׾����׾ʾ���������������������������4�@�M�Y�]�Y�M�4��čā�k�j�xČđĠ�������������9���ĿĦč�H�@�4�'�%�'�4�M�f�������������r�f�Y�H�������������������ûлػԻл̻û����������������¿Ŀѿݿ�����������ݿѿĿ����_�_�S�R�S�_�d�l�x���������������x�l�_�_��������������"������������������6�6�+�6�B�O�P�O�G�B�6�6�6�6�6�6�6�6�6�6FF	FFF%F1F7FJFVFcFoFsFtFdFVFKF=F1F$F���������ĿϿϿȿĿ�������������������������������"�)�6�7�;�6�)���������������������ûлػԻлû������������������������ûܻ�����������ܻû��������������������������������:�.����	��!�.�G�S�������}�s�g�S�G�:�����ݽҽݽ������������
������������������������������������������������~�r�j�p�r�~������������������������D�D�D�D�D�D�EEEEEEED�D�D�D�D�D�D�E*EE*E4E7ECEPETETEPECE7E*E*E*E*E*E*E*E*�G�<�3�2�<�H�I�U�e�n�zÍÒÇÃ�z�o�a�U�G Y Y I T V R 2 d @ A  5 % 0 Q -  V U * 5 r F ( ^ 6 ; X B j ( 2 E   .  \  < W 8 { ( C , G z G . P  F m M V l H G P P � 0 E �  e 7 4 y - K o I    l  �  4  �  ;  �    �  �  B  �  �  d  �  D  a  �  w    �  _  x  �  H  f  K  �  �  �  �  P  "  �    �  x  �  �  W  Y  s  d  �  u    M  k    7  W  �  �    <  �  �  a  �  �  4    [  �  "  �  %  5    �  �  �  z  ;  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  z  ^  B  &    �  �  �  �  �  �  �  e  <  	  �  ~  �  �  8  �  �  �  �  �  �  z  c  @    �  �  �  a  +  �  �  �  ]  (  A  F  L  N  P  N  K  P  Y  T  C  2  !    �  �  �  �  �  �  E  B  @  :  .  #      �  �  �  �  �  �  �  �  v  ^  F  -  �  �          �  �  �  �  u  E    �  �    �    �  -  P  q  ~    l  G    �  ~    �  u  "  �  j  �  }  �     �  �  �  �  �  �  �  �  �  e  (  �  u  
  �    �  :  �  F  �    %  2  6  5  3  -  '          �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  w  e  P  ;  &    �  �  �  �  \  k  u  z  �  �  �  �  w  X  3    �  �  A  �  �  K  �  o  X  m  c  j  s  }  �  �  �  v  V  3  	  �  }     �  �  �  5  +  <  P  g  t  w  l  _  ]  i  i  c  K  '  �  �  �  /  �  e  �  "  ?  R  Z  T  C  )     �  �  ;  �  �  ,  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  u  Y  3     �   �   �  /  O  [  ^  \  S  H  ;  +      �  �  �  s  D    �  �      i  �  �  �  �  �  �  �  �  q  N    �  1  �  �  Z  �  �  �  �  +  I  Y  f  h  ^  F    �  �  q    �  %  g  �  G  �     )  Q  s  �  �  �  �  �  �  {  S    �  �  ,  �  �  ,  �  �  5  a  j  i  e  l  m  T  )  �  �  �  a  +  �  Y  �  �  �  �    >  W  ]  S  K  C  0      �  �  �  2  �  
  |  �   �             �  �  �  �  �  �  �  �  t  _  J    �  �  M  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  }  a  @    �  �  �  i  :  	  �  �  o  �  �  �  �  �  {  f  Q  <  '    �  �  �  �  m  E    �  �  q  y  �  �  }  x  p  f  W  K  :     �  �  �  q  6  �  |   �  P  {  �  �  �      .  1  1  -  "    �  �  a  �  �  A  �  d  j  o  u  s  o  l  U  5    �  �  �  �  �  v  a  K  4    �  �  �  �  �  �  �  �  �  �    A  �  i  3  �  =  �  �    �  �  �  �  �  �  {  d  K  .    �  �  �  f  3  �  �  p  (  /  :  C  6  )          �  �  �  �  G    �  �  �  |  R              �  �  �  �  �  �  �  |  Y  -  �  �  {  +   �   �   �   �   �   �   �   �   �   �   �   �   �   �   �   �   �   �   {   q  g  g  �  �  �  �  �  �  �  ~  ?  �  �  2  �  c    ]  �  �  o  k  l  n  b  R  @  ,    �  �  �  �  �  q  S  6  !      m  �  �  �  �  �  �  �  ~  M    �  �  *  �    i  Y  �  Y  ;  g  o  �  �  �  �  �  �  |  t  g  O  2    �  z    �   �  �  �  �  �  �  �  o  V  9    �  �  T  �  �  �  �  �  �  f  =  b  �  �  �  �  �  �  �  g  ,  �  �  C  �  �  F    �  �  	          �  �  �  �  �  �  �  M    �  �  a     �   �  �  �  �  �  �  �  �  �  �  �  �  �  �  |  s  k  c  [  S  J  /      �  �  �  �  c  5    �  �  v  D    �  �  v  �  =  X  Q  I  3    �  �  �  �  �  v  c  T  A  !     �  �  �  ]  �  �  �  �  �  �  �  �  �  �  �  |  i  U  A  .      �  �  �  �  �  �  �  �  �  �  �  �  l  D    �  c  �  i  �  u  C          �  �  �  �  �  [  4  
  �  �  c    �  (  �  �  �  �  �  �  �  �  �  ~  ]  9    �  �  �  X    �  L  �  h  S  D  3    	  �       �  �  �  �  f  @    �  �  �  �  K    4  4  .  (    �  �  �  �  j  9    �  {  #  �  9  �  �  �  #  `  �  �  �  �  �  �  �  �  {  >  �  �  ?  �  �  �  �  �  �  �  �  �  �  �  p  R  ,    �  �  h  A  (  $  &  �  �  _  c  d  a  W  K  9  #  
  �  �  �  ~  P    �  �  ]    �  �  �  �  �  �  �  �  �  }  m  ]  M  6      �  �    @  a  �  �  �  �  �  �  �  �  �  �  �  �  �  �  N  
  �  �  	  u  u  k  G  0    �  w  I  �  �  k  0    �  �  �    S  �  4  6  �  �  a  /  �  �  �  F  
�  
�  
  	  �  +  �  �  �  �  �  R  @  /  "      $  4  9  #  �  �  �  {  =  �  �  r  )  �  �  �  �  �  �  o  V  <  +       �  �  �  y  A  �  �  e    R  i  i  [  J  3       �  �  �  �  p  L    �  �  \    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  ~  r  ^  $  �  �  7  '      �  �  �  �  �  �  j  P  2    �  �  �  �  w  W  �    L    �  �  c  !  �  �  R    �  r  "  �  �    �  �  �  �  �  �  y  a  G  +    �  �  �  �  j  F  #  �  �  �  V  �    )  C  ;  (    �  �  �  �  \  0    �  �      n  �  1        �  �  �  �  �  �  �  b    �  >  �  �  V     �  d  T  >  &      �  �  �  �  �  l  G    �  �  �  �  Q   �  �  �  �  �  �  �  �  �  g  M  2    �  �  �  \  
  �  �  ]  �  �  �  �  r  S  0    �  �  M  	  �  u    �  Y  �  �  u  x  j  ^  i  �  �  �  �  �  �  j  N  1    �  �  {  ?    �  �  �  l  V  6    �  �  �  �  �  �  k  L  +  
   �   �   �   �  y  g  Q  9    �  �  �  �  v  S  0    �  �  �  y  l  h  m  +  /  #    �  �  s  9    �  z  3  �  �  R    �  W  �  �    �  �  �  �  �  �  p  Q  0    �  �  j  -  �  �  �  [  '  a  U  7    �  �  �  B  �  �  \    �    �  2  �  �  r  J